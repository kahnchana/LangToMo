"""
This file trains a action policy network conditioned on optical flow inputs (and other additional inputs).
"""

import argparse
import dataclasses
import os

import accelerate
import torch
import tqdm
from diffusers import optimization

from src.dataset import calvin_dataset
from src.model import vit


@dataclasses.dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 4
    eval_batch_size: int = 4  # how many images to sample during evaluation
    num_gpu: int = 4
    data_root: str = "/home/kanchana/data/calvin/task_ABC_D"
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    num_epochs: int = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", type=int, default=4)
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


def evaluate(config, epoch, dataloader, model, split="train", eval_steps=-1):
    # Calculate loss over the dataset.
    count = 0
    total_loss = 0
    for idx, batch in enumerate(dataloader):
        flow = batch["flow"]
        relative_action = batch["relative_action"]

        with torch.no_grad():
            pred_action = model(flow)

        loss = torch.mean((pred_action - relative_action) ** 2)
        total_loss += loss
        count += 1

        if eval_steps > 0 and idx > eval_steps:
            break

    total_loss /= count

    return {f"{split}/loss": total_loss}


def train_loop(config, model, optimizer, train_dataloader, val_dataloader, lr_scheduler):
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
        accelerator.init_trackers(
            "LangToMo",
            init_kwargs={"wandb": {"config": dataclasses.asdict(config), "group": "policy_net"}},
        )

    # Prepare everything, no specific inputs order to remember for prepare function.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # if step > 20:  # Debug Code
            #     break

            # image = batch["image"]
            flow = batch["flow"]
            relative_action = batch["relative_action"]

            with accelerator.accumulate(model):
                pred = model(flow)
                loss = torch.nn.functional.mse_loss(pred, relative_action)
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

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                train_set_stats = evaluate(config, epoch, train_dataloader, model, split="train", eval_steps=1000)
                val_set_stats = evaluate(config, epoch, val_dataloader, model, split="val")
                log_data = {
                    "epoch": epoch,
                    **train_set_stats,
                    **val_set_stats,
                }
                accelerator.log(log_data, step=global_step)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model.module.save_pretrained(config.output_dir)


if __name__ == "__main__":
    config = TrainingConfig()
    opts = parse_args()
    config = update_config_with_args(config, opts)

    CAPTION_FILE = f"{config.data_root}/captions.json"
    ACTION_FILE = f"{config.data_root}/relative_actions.json"

    transform, _ = calvin_dataset.get_joint_transforms(image_size=(128, 128))  # Convert to tensor and resize only.
    train_root = f"{config.data_root}/robot_training"
    val_root = f"{config.data_root}/robot_validation"

    train_captions = os.path.join(train_root, "captions.json")
    val_captions = os.path.join(val_root, "captions.json")
    train_actions = os.path.join(train_root, "relative_actions.json")
    val_actions = os.path.join(val_root, "relative_actions.json")

    train_dataset = calvin_dataset.RobotTrainingDataset(
        train_root, train_captions, train_actions, transform=transform, include_captions=False, include_actions=True
    )
    val_dataset = calvin_dataset.RobotTrainingDataset(
        val_root, val_captions, val_actions, transform=transform, include_captions=False, include_actions=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_gpu
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_gpu
    )

    model = vit.get_vit_tiny(image_size=128, patch_size=16, in_channels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optimization.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    accelerate.notebook_launcher(train_loop, args, num_processes=config.num_gpu)
