import argparse
import os

import torch

from src.dataset import calvin_dataset
from src.model import diffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opts = parse_args()

    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    # Load dataset.
    image_size = (128, 128)
    image_transform, flow_transform = calvin_dataset.get_transforms(image_size)
    dataset = calvin_dataset.RobotTrainingDataset(
        DATA_ROOT, CAPTION_FILE, transform=image_transform, target_transform=flow_transform
    )

    # Setup dataloader.
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)

    # Setup model.
    unet_model = diffusion.get_conditional_unet(image_size)
