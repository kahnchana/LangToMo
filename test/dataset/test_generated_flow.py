import json

import einops
import numpy as np

from src.dataset import flow_utils

# Uncomment one to select split for visualization.
SPLIT_OPTIONS = ["training", "validation"]
TASK_OPTIONS = ["D_D", "ABC_D"]
SPLIT = SPLIT_OPTIONS[1]
TASK = TASK_OPTIONS[1]

DATA_ROOT = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT}"

episode_idx = 10
cur_name = f"eps_{episode_idx:05d}"
datum_file = f"{DATA_ROOT}/{cur_name}.npz"
datum = np.load(datum_file)

caption_file = f"{DATA_ROOT}/captions.json"
caption_data = json.load(open(caption_file, "r"))
caption = caption_data[cur_name]

normalizer = flow_utils.FlowNormalizer(200, 200)

images = einops.rearrange((datum["image"] * 255).astype(np.uint8), "b c h w -> b h w c")
flow = einops.rearrange(datum["flow"], "b c h w -> b h w c")
flow = normalizer.unnormalize(flow)

cidx = 7
flow_utils.visualize_flow_vectors(images[cidx], flow[cidx], step=4, save_path=None, title=caption)

print(datum["image"].min(), datum["image"].max(), datum["image"].shape)
print(datum["flow"].min(), datum["flow"].max(), datum["flow"].shape)
