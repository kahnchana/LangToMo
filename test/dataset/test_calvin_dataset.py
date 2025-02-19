import os

import einops
import numpy as np
import torch
from PIL import Image

from src.dataset import calvin_dataset, flow_utils

# Example usage
if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    image_size = (128, 128)
    mask_opt = {"mask_percentage": 0.5, "patch_size": 16}
    crop_ratio = 0.7
    mask_crop_ratio = 0.5
    train_transform, val_transform = calvin_dataset.get_joint_transforms(
        image_size=image_size,
        add_color_jitter=False,
        mask_args=mask_opt,
        crop_ratio=crop_ratio,
        mask_crop_ratio=mask_crop_ratio,
    )
    dataset = calvin_dataset.RobotTrainingDataset(
        DATA_ROOT, CAPTION_FILE, transform=train_transform, include_captions=True
    )
    dataset_nt = calvin_dataset.RobotTrainingDataset(
        DATA_ROOT, CAPTION_FILE, transform=val_transform, include_captions=True
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    dataloader_nt = torch.utils.data.DataLoader(dataset_nt, batch_size=8, shuffle=False)

    for sample, sample_nt in zip(dataloader, dataloader_nt):
        print(
            f"Image shape: {sample['image'].shape}, "
            f"Flow shape: {sample['flow'].shape}, "
            f"Caption: {sample['caption']}, "
            f"Caption Emb: {sample['caption_emb'].shape}, "
        )
        print(
            f"Image shape: {sample_nt['image'].shape}, "
            f"Flow shape: {sample_nt['flow'].shape}, "
            f"Caption: {sample_nt['caption']}, "
            f"Caption Emb: {sample_nt['caption_emb'].shape}, "
        )
        break

    normalizer = flow_utils.FlowNormalizer(*sample["image"].shape[2:])

    cidx = 1
    images = einops.rearrange((sample["image"].numpy() * 255).astype(np.uint8), "b c h w -> b h w c")
    flow = einops.rearrange(sample["flow"].numpy(), "b c h w -> b h w c")
    flow = normalizer.unnormalize(flow)
    vis_aug = flow_utils.visualize_flow_vectors_as_PIL(images[cidx], flow[cidx], step=4)

    images = einops.rearrange((sample_nt["image"].numpy() * 255).astype(np.uint8), "b c h w -> b h w c")
    flow = einops.rearrange(sample_nt["flow"].numpy(), "b c h w -> b h w c")
    flow = normalizer.unnormalize(flow)
    vis_orig = flow_utils.visualize_flow_vectors_as_PIL(images[cidx], flow[cidx], step=4)

    vis = Image.new("RGB", (vis_aug.width * 2, vis_aug.height))
    vis.paste(vis_aug, (0, 0))
    vis.paste(vis_orig, (vis_aug.width, 0))
    vis
