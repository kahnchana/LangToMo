import os

import einops
import numpy as np
import torch

from src.dataset import calvin_dataset, flow_utils

# Example usage
if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    dataset = calvin_dataset.RobotTrainingDataset(DATA_ROOT, CAPTION_FILE, include_captions=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    for sample in dataloader:
        print(
            f"Image shape: {sample['image'].shape}, "
            f"Flow shape: {sample['flow'].shape}, "
            f"Caption: {sample['caption']}, "
            f"Caption Emb: {sample['caption_emb'].shape}, "
        )
        break

    normalizer = flow_utils.FlowNormalizer(*sample["image"].shape[2:])

    images = einops.rearrange((sample["image"].numpy() * 255).astype(np.uint8), "b c h w -> b h w c")
    flow = einops.rearrange(sample["flow"].numpy(), "b c h w -> b h w c")
    flow = normalizer.unnormalize(flow)
    cidx = 1
    flow_utils.visualize_flow_vectors(images[cidx], flow[cidx], step=4, save_path=None)
