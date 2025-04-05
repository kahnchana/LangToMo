import argparse
import dataclasses
import os
import random
from functools import partial

import accelerate
import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from torchvision.models import optical_flow
from tqdm.auto import tqdm

import wandb
from src import inference
from src.dataset import calvin_dataset, flow_utils, metaworld
from src.dataset import openx_trajectory_dataset as openx_traj
from src.model import diffusion


@dataclasses.dataclass
class TrainingConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 32  # how many images to sample during evaluation
    num_gpu: int = 4
    port: int = 8807
    dataset: str = "calvin"
    sub_datasets: list = dataclasses.field(default_factory=lambda: ["calvin"])
    val_sub_datasets: list = dataclasses.field(default_factory=lambda: ["bridge"])
    learning_rate: float = 1e-4
    num_train_steps: int = 300_000
    evaluate_interval_steps: int = 2_500
    save_interval_steps: int = 5_000
    gradient_accumulation_steps: int = 1
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed: int = 0
    image_size: int = 128  # the generated image resolution
    mask_ratio: float = 0.50
    mask_patch: int = 16
    crop_ratio: float = 0.7
    mask_crop_ratio: float = 0.5
    prev_flow: bool = False
    temporal_unet: bool = False
    num_frames: int = 8
    pretrained: str = None
    resume: bool = False
    wandb_id: str = None
    output_dir: str = "experiments/test_001"  # the model name locally
    debug: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="calvin")
    parser.add_argument("--sub-datasets", type=str, nargs="+", default=["bridge"], help="List of sub-datasets")
    parser.add_argument("--val-sub-datasets", type=str, nargs="+", default=["bridge"], help="List of val sub-datasets")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--steps", type=int, default=300_000, dest="num_train_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--mask-patch", type=int, default=None)
    parser.add_argument("--crop-ratio", type=float, default=None)
    parser.add_argument("--mask-crop-ratio", type=float, default=None)
    parser.add_argument("--prev-flow", action="store_true", default=False)
    parser.add_argument("--temporal", action="store_true", default=False, dest="temporal_unet")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="test_002")
    parser.add_argument("--port", type=int, default=8807)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def update_config_with_args(config: TrainingConfig, opts: argparse.Namespace) -> TrainingConfig:
    config_dict = config.__dict__.copy()
    for key, value in vars(opts).items():
        if key in config_dict:
            config_dict[key] = value
        else:
            raise ValueError(f"Argument: {key} not found in config.")

    return dataclasses.replace(config, **config_dict)


def get_dataset(config, accelerator=None):
    # Load dataset.
    if config.dataset == "openx":
        traj_len = 3 if config.prev_flow else config.num_frames + 1
        sub_datasets = config.sub_datasets
        data_root = "/home/kanchana/data/openx"
        traj_dataset = openx_traj.OpenXTrajectoryDataset(
            root_dir=data_root, datasets=sub_datasets, split="train", trajectory_length=traj_len, infinite_repeat=True
        )
        dataset_name = sub_datasets[0]  # TODO: add support for multiple datasets
        dataset_size = traj_dataset.dataset_sizes[dataset_name]
        dataset = openx_traj.DatasetIterator(
            traj_dataset.dataset_dict[dataset_name], length=dataset_size, get_command=False
        )

        val_name = config.val_sub_datasets[0]  # TODO: add support for multiple datasets
        val_dataset = openx_traj.OpenXTrajectoryDataset(
            root_dir=data_root, datasets=[val_name], split="train", trajectory_length=traj_len
        )
        val_size = val_dataset.dataset_sizes[val_name]
        val_dataset = openx_traj.DatasetIterator(val_dataset.dataset_dict[val_name], length=val_size, get_command=False)
        if accelerator is not None:
            # Ensure correct process management for sharded dataset.
            train_dataloader = accelerate.data_loader.IterableDatasetShard(
                dataset,
                batch_size=config.train_batch_size,
                num_processes=accelerator.num_processes,
                process_index=accelerator.process_index,
            )
            val_dataloader = accelerate.data_loader.IterableDatasetShard(
                val_dataset,
                batch_size=config.eval_batch_size,
                num_processes=accelerator.num_processes,
                process_index=accelerator.process_index,
            )
            # Still necessary for actual batching operations.
            train_dataloader = torch.utils.data.DataLoader(
                train_dataloader, batch_size=config.train_batch_size, num_workers=0
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataloader, batch_size=config.eval_batch_size, num_workers=0
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.train_batch_size, num_workers=config.num_gpu
            )
            val_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.eval_batch_size, num_workers=config.num_gpu
            )

    elif config.dataset == "metaworld":
        DATA_ROOT = "/home/kanchana/data/metaworld"
        target_size = (config.image_size, config.image_size)
        traj_len = 3 if config.prev_flow else config.num_frames + 1
        dataset = metaworld.MetaworldDataset(
            DATA_ROOT, target_size=target_size, captions=False, sample_per_seq=traj_len, randomcrop=False
        )
        train_dataset = metaworld.InfiniteWrapper(dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_gpu
        )

        val_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_gpu
        )

    else:
        SPLIT_OPTIONS = ["training", "validation"]
        if config.dataset == "calvin":
            TASK_OPTIONS = ["D_D", "ABC_D"]
            TASK = TASK_OPTIONS[1]
            train_data_root = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT_OPTIONS[0]}"
            val_data_root = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT_OPTIONS[1]}"
        else:
            data_root = "/home/kanchana/data/ssv2_flow"
            train_data_root = f"{data_root}/{SPLIT_OPTIONS[0]}"
            val_data_root = f"{data_root}/{SPLIT_OPTIONS[1]}"

        train_captions = os.path.join(train_data_root, "captions.json")
        val_captions = os.path.join(val_data_root, "captions.json")

        image_size = (config.image_size, config.image_size)
        if config.mask_ratio is None:
            mask_opt = None
        else:
            mask_opt = {"mask_percentage": config.mask_ratio, "patch_size": config.mask_patch}
        train_transform, val_transform = calvin_dataset.get_joint_transforms(
            image_size=image_size,
            add_color_jitter=False,
            mask_args=mask_opt,
            crop_ratio=config.crop_ratio,
            mask_crop_ratio=config.mask_crop_ratio,
        )
        train_dataset = calvin_dataset.RobotTrainingDataset(
            train_data_root, train_captions, transform=train_transform, include_prev_flow=config.prev_flow
        )
        val_dataset = calvin_dataset.RobotTrainingDataset(
            val_data_root, val_captions, transform=val_transform, include_prev_flow=config.prev_flow
        )

        # Setup dataloader.
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_gpu
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_gpu
        )

    return train_dataloader, val_dataloader


def generate_flow_online(image_condition, flow_model, flow_transform, get_all_flow=False):
    flow_input, _ = flow_transform(image_condition, image_condition)
    start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
    end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]
    flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=image_condition.shape[0])
    flow_dim = flow_tensor.shape[-1]  # Image size is always square.
    normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
    if get_all_flow:
        return normalized_flow_tensor, None
    clean_flow, prev_flow = normalized_flow_tensor[:, 1], normalized_flow_tensor[:, 0]
    return clean_flow, prev_flow


def evaluate(dataloader, model, flow_model, flow_transform, config, split="train"):
    RUN_COUNT = 25
    if config.debug:
        RUN_COUNT = 3
    count = 0
    total_loss = 0
    for idx, batch in tqdm(enumerate(dataloader), total=RUN_COUNT):
        image_input = batch["observation"]
        text_cond = batch["caption_embedding"]
        clean_flow, prev_flow = generate_flow_online(image_input, flow_model, flow_transform, config.temporal_unet)
        if config.temporal_unet:
            clean_flow = einops.rearrange(clean_flow, "b t c h w -> b c t h w")
            vis_cond = einops.rearrange(image_input[:, :1], "b t c h w -> b c t h w")
            vis_cond = vis_cond.repeat(1, 1, clean_flow.shape[2], 1, 1)  # repeat first image T times
            start_flow = torch.randn(clean_flow.shape, device=clean_flow.device)
        else:
            image_cond = image_input[:, 1]
            start_flow = torch.randn(clean_flow.shape, device=clean_flow.device)  # random noise
            if config.prev_flow:
                vis_cond = torch.cat([image_cond, prev_flow], dim=1)
            else:
                vis_cond = image_cond

        generated_flow = inference.run_inference(model, start_flow, vis_cond, text_cond, num_inference_steps=25)
        with torch.no_grad():
            loss = torch.mean((generated_flow - clean_flow) ** 2)

        total_loss += loss.item()
        count += 1

        if True:  # split == "train":
            if idx >= RUN_COUNT:
                break

    total_loss /= count

    # Generate visualizations.
    if config.temporal_unet:
        vis_count = config.num_frames
        vis_images = einops.rearrange((image_input[0, :-1].cpu().numpy() * 255).astype(np.uint8), "t c h w -> t h w c")

        normalizer = flow_utils.FlowNormalizer(config.image_size, config.image_size)
        vis_pred = einops.rearrange(generated_flow[0].cpu().numpy(), "c t h w -> t h w c")
        vis_pred = normalizer.unnormalize(vis_pred)

        vis_gt = einops.rearrange(clean_flow[0].cpu().numpy(), "c t h w -> t h w c")
        vis_gt = normalizer.unnormalize(vis_gt)

    else:
        vis_count = 4
        vis_images = image_cond[:vis_count]
        vis_images = einops.rearrange((vis_images.cpu().numpy() * 255).astype(np.uint8), "b c h w -> b h w c")

        normalizer = flow_utils.FlowNormalizer(config.image_size, config.image_size)
        vis_pred = einops.rearrange(generated_flow[:vis_count].cpu().numpy(), "b c h w -> b h w c")
        vis_pred = normalizer.unnormalize(vis_pred)

        vis_gt = einops.rearrange(clean_flow[:vis_count].cpu().numpy(), "b c h w -> b h w c")
        vis_gt = normalizer.unnormalize(vis_gt)

    vis_gt_flow = [
        flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_gt[i], step=4, title="Ground Truth Optical Flow")
        for i in range(vis_count)
    ]
    vis_pred_flow = [
        flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_pred[i], step=4, title="Predicted Optical Flow")
        for i in range(vis_count)
    ]

    vis_joint = [make_image_grid([x, y], rows=1, cols=2) for x, y in zip(vis_gt_flow, vis_pred_flow)]

    return {f"{split}/images": [wandb.Image(x) for x in vis_joint], f"{split}_loss": total_loss}


def train_loop(config):
    # Set seed.
    accelerate.utils.set_seed(config.seed)

    # Initialize accelerator and tensorboard logging
    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(f"{config.output_dir}/model", exist_ok=True)
            os.makedirs(f"{config.output_dir}/train_state", exist_ok=True)
        log_init_args = {"wandb": {"config": dataclasses.asdict(config)}}
        if config.resume:
            log_init_args["wandb"]["resume"] = "must"
            log_init_args["wandb"]["id"] = config.wandb_id
        accelerator.init_trackers(
            "LangToMo",
            init_kwargs=log_init_args,
        )

    # Debug code.
    if config.debug:
        config.evaluate_interval_steps = 10

    # Load Data.
    train_dataloader, val_dataloader = get_dataset(config, accelerator=accelerator)

    # Setup model.
    if config.resume:
        config.pretrained = f"{config.output_dir}/model"

    if not config.temporal_unet:
        in_channels = 7 if config.prev_flow else 5
        model = diffusion.get_conditional_unet(
            config.image_size, config.pretrained, in_channels=in_channels, out_channels=2, condition_dim=512
        )
    else:
        model = diffusion.get_conditional_unet_3d(
            image_size=config.image_size, pretrained=config.pretrained, condition_dim=512
        )

    # Setup flow model.
    flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
    flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    # Noise scheduler, optimizer and LR scheduler.
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(config.lr_warmup_steps * config.num_gpu),
        num_training_steps=(config.num_train_steps * config.num_gpu),
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler, flow_model, flow_transform = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler, flow_model, flow_transform
    )

    # Define evaluation function.
    eval_func = partial(evaluate, flow_model=flow_model, flow_transform=flow_transform, config=config)

    data_iter = iter(train_dataloader)
    step = 0
    if config.resume:
        # Load last step.
        with open(f"{config.output_dir}/last_step.txt", "r") as f:
            step = int(f.read().strip())

        # Load train state.
        accelerator.load_state(f"{config.output_dir}/train_state")

    progress_bar = tqdm(initial=step, total=config.num_train_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    while step < config.num_train_steps:
        # Handle infinite dataloader.
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch ended, restart iterator
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        except Exception as e:
            print(e)
            continue

        # Load data.
        image_inputs = batch["observation"]
        text_condition = batch["caption_embedding"]

        # Generate flow online.
        clean_flow, prev_flow = generate_flow_online(image_inputs, flow_model, flow_transform, config.temporal_unet)

        bs = clean_flow.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_flow.device, dtype=torch.int64
        )

        if config.temporal_unet:
            clean_flow = einops.rearrange(clean_flow, "b t c h w -> b c t h w")
            image_condition = einops.rearrange(image_inputs[:, :1], "b t c h w -> b c t h w")
            image_condition = image_condition.repeat(1, 1, clean_flow.shape[2], 1, 1)  # repeat first image T times
            noise = torch.randn(clean_flow.shape, device=clean_flow.device)
            noisy_flow = noise_scheduler.add_noise(clean_flow, noise, timesteps)
            model_inputs = torch.concat([noisy_flow, image_condition], dim=1)

        else:
            image_condition = image_inputs[:, 1]  # Select the second image in sequence.
            noise = torch.randn(clean_flow.shape, device=clean_flow.device)
            noisy_flow = noise_scheduler.add_noise(clean_flow, noise, timesteps)
            model_inputs = [noisy_flow, image_condition]
            if config.prev_flow:
                pick = random.random() < 0.5  # TODO: make this a hyperparameter
                if pick:
                    prev_flow = torch.ones_like(prev_flow) * 0.5
                model_inputs.append(prev_flow)
            model_inputs = torch.concat(model_inputs, dim=1)

        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(model_inputs, timesteps, text_condition, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": step}
        progress_bar.set_postfix(**logs)

        # Run evaluation and visualization.
        eval_log_data = {}
        if (step + 1) % config.evaluate_interval_steps == 0 or step == config.num_train_steps:
            if config.dataset in ["metaworld"]:
                train_logs = {}
            else:
                train_logs = eval_func(train_dataloader, model, split="train")
            val_logs = eval_func(val_dataloader, model, split="val")
            eval_log_data = {**train_logs, **val_logs}
            if accelerator.is_main_process:
                if config.debug:
                    print(eval_log_data)

        # Log to W&B
        accelerator.log({**logs, **eval_log_data}, step=step)

        # Save the model.
        if accelerator.is_main_process:
            if (step + 1) % config.save_interval_steps == 0 or step == config.num_train_steps:
                model.module.save_pretrained(f"{config.output_dir}/model")
                accelerator.save_state(f"{config.output_dir}/train_state")
                with open(f"{config.output_dir}/last_step.txt", "w") as f:
                    f.write(str(step))

        # End training.
        step += 1
        if step >= config.num_train_steps:
            break


if __name__ == "__main__":
    config = TrainingConfig()
    opts = parse_args()
    config = update_config_with_args(config, opts)

    # Call training.
    train_args = (config,)
    accelerate.notebook_launcher(train_loop, train_args, num_processes=config.num_gpu, use_port=config.port)
