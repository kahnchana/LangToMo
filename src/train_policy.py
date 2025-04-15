"""
This file trains a action policy network conditioned on optical flow inputs (and other additional inputs).
"""

import argparse
import dataclasses
import os
from functools import partial

import accelerate
import torch
import torch.nn.functional as F
import tqdm
from diffusers import optimization
from torchvision.models import optical_flow

from src.dataset import supervised_dataset
from src.model import vit


@dataclasses.dataclass
class TrainingConfig:
    num_gpu: int = 4
    dataset: str = "metaworld"
    train_batch_size: int = 32
    eval_batch_size: int = 32  # how many images to sample during evaluation
    learning_rate: float = 1e-4
    num_train_steps: int = 10_000
    evaluate_interval_steps: int = 2_500
    save_interval_steps: int = 5_000
    gradient_accumulation_steps: int = 1
    lr_warmup_steps: int = 100
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    image_size: int = 128  # the generated image resolution
    pretrained: str = None
    resume: bool = False
    wandb_id: str = None
    output_dir: str = "experiments/mw_policy_001"  # the model name locally
    seed: int = 0
    port: int = 8807
    debug: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="calvin")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, dest="learning_rate")
    parser.add_argument("--steps", type=int, default=10_000, dest="num_train_steps")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="test_002")
    parser.add_argument("--seed", type=int, default=0)
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


def generate_flow_online(config, image_condition, flow_model, flow_transform):
    flow_input, _ = flow_transform(image_condition, image_condition)  # b, t, c, h, w
    start_im = flow_input[:, 0]  # frame t
    end_im = flow_input[:, 2]  # frame t+k
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]  # b, c, h, w
    image_size = config.image_size
    scale = 256 / image_size  # Hard Coded.
    flow_tensor = F.interpolate(flow_tensor, size=(image_size, image_size), mode="bilinear", align_corners=True)
    flow_tensor = flow_tensor / scale

    flow_dim = flow_tensor.shape[-1]  # Image size is always square.
    normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)

    return normalized_flow_tensor


def get_dataset(config):
    if config.dataset == "metaworld":
        DATA_ROOT = "/home/kanchana/data/metaworld/mw_traj"
        train_dir = f"{DATA_ROOT}/door-open-v2-goal-observable"
        train_dataset = supervised_dataset.SupervisedDataset(root_dir=train_dir, k=10)
        val_dir = f"{DATA_ROOT}/door-close-v2-goal-observable"
        val_dataset = supervised_dataset.SupervisedDataset(root_dir=val_dir, k=10)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=4
        )
    else:
        raise NotImplementedError(f"Dataset: {config.dataset} not implemented.")

    return train_dataloader, val_dataloader


def evaluate(dataloader, model, flow_model, flow_transform, config, split="train"):
    # Calculate loss over the dataset.
    eval_steps = -1
    if config.debug:
        eval_steps = 3  # Debug code
    count = 0
    total_loss = 0
    total_steps = eval_steps if eval_steps > 0 else len(dataloader)
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=total_steps):
        clean_flow = generate_flow_online(config, batch["observation_256"], flow_model, flow_transform)
        im_t, im_tpi = batch["observation"][:, 0], batch["observation"][:, 1]  # b, c, h, w
        model_input = torch.cat([im_t, im_tpi, clean_flow], dim=1)
        target_action = batch["action"]  # b, 4
        with torch.no_grad():
            pred_action = model(model_input, return_dict=False)[0]
            loss = torch.mean((pred_action - target_action) ** 2)

        total_loss += loss.item()
        count += 1

        if eval_steps > 0 and idx > eval_steps:
            break

    total_loss /= count

    return {f"{split}/loss": total_loss}


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
        log_init_args = {"wandb": {"config": dataclasses.asdict(config), "group": "policy_net"}}
        if config.resume:
            log_init_args["wandb"]["resume"] = "must"
            log_init_args["wandb"]["id"] = config.wandb_id
        accelerator.init_trackers("LangToMo", init_kwargs=log_init_args)

    # Debug code.
    if config.debug:
        config.evaluate_interval_steps = 10

    # Load Data.
    train_dataloader, val_dataloader = get_dataset(config)

    # Setup model.
    if config.resume:
        config.pretrained = f"{config.output_dir}/model"

    channels = 8
    model = vit.get_vit_tiny_hf(image_size=128, patch_size=16, in_channels=channels, action_space=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(config.lr_warmup_steps * config.num_gpu),
        num_training_steps=(config.num_train_steps * config.num_gpu),
    )

    # Setup flow model.
    flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
    flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    # Prepare everything, no specific inputs order to remember for prepare function.
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

    progress_bar = tqdm.tqdm(initial=step, total=config.num_train_steps, disable=not accelerator.is_local_main_process)
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

        im_t, im_tpi = batch["observation"][:, 0], batch["observation"][:, 1]  # b, c, h, w
        target_action = batch["action"]  # b, 4

        # Generate flow online.
        clean_flow = generate_flow_online(config, batch["observation_256"], flow_model, flow_transform)  # b, c, h, w
        model_input = torch.cat([im_t, im_tpi, clean_flow], dim=1)

        with accelerator.accumulate(model):
            pred_action = model(model_input, return_dict=False)[0]
            loss = torch.nn.functional.mse_loss(pred_action, target_action)
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

    args = (config,)
    accelerate.notebook_launcher(train_loop, args, num_processes=config.num_gpu)
