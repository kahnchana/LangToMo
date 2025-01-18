import argparse
import os
from dataclasses import asdict, dataclass

import einops
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, notebook_launcher
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

import wandb
from src import inference
from src.dataset import calvin_dataset, flow_utils
from src.model import diffusion


@dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 32
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 500
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 10
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = "experiments/test_001"  # the model name locally and on the HF Hub
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 0


def evaluate(config, epoch, dataloader, model):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    for batch in dataloader:
        clean_flow = batch["flow"]
        image_cond = batch["image"]
        text_cond = batch["caption_emb"]
        start_flow = torch.randn(clean_flow.shape, device=clean_flow.device)  # random noise

        generated_flow = inference.run_inference(model, start_flow, image_cond, text_cond, num_inference_steps=50)

        break  # get single example for visualization

    # Make a grid out of the images.
    vis_count = 4
    vis_images = einops.rearrange((image_cond[:vis_count].cpu().numpy() * 255).astype(np.uint8), "b c h w -> b h w c")

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

    # Save the images.
    wandb.log(
        {
            "epoch": epoch,
            "images": [wandb.Image(x) for x in vis_joint],
        }
    )


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(
            "train_example",
            init_kwargs={"wandb": {"config": asdict(config)}},
        )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # if step > 20:  # Debug Code
            #     break

            # Load data.
            clean_flow = batch["flow"]
            image_condition = batch["image"]
            text_condition = batch["caption_emb"]

            # Sample noise to add to the clean flow.
            noise = torch.randn(clean_flow.shape, device=clean_flow.device)
            bs = clean_flow.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_flow.device, dtype=torch.int64
            )

            # Add noise to the clean flow according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_flow = noise_scheduler.add_noise(clean_flow, noise, timesteps)
            model_inputs = torch.concat([noisy_flow, image_condition], dim=1)

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
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, train_dataloader, model)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model.module.save_pretrained(config.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="test_002")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = TrainingConfig()
    opts = parse_args()
    config.output_dir = f"experiments/{opts.output_dir}"

    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    # Load dataset.
    image_size = (128, 128)
    image_transform, flow_transform = calvin_dataset.get_transforms(image_size)
    dataset = calvin_dataset.RobotTrainingDataset(
        DATA_ROOT, CAPTION_FILE, transform=image_transform, target_transform=flow_transform
    )

    # Setup dataloader.
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False)

    # Setup model.
    unet_model = diffusion.get_conditional_unet(config.image_size)

    # Noise scheduler, optimizer and LR scheduler.
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Call training.
    args = (config, unet_model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=opts.num_gpu)
