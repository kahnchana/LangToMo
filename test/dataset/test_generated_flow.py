import json

import einops
import numpy as np

from src.dataset import calvin

DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"

episode_idx = 10
cur_name = f"eps_{episode_idx:05d}"
datum_file = f"{DATA_ROOT}/{cur_name}.npz"
datum = np.load(datum_file)

caption_file = f"{DATA_ROOT}/captions.json"
caption_data = json.load(open(caption_file, "r"))
caption = caption_data[cur_name]

images = einops.rearrange(calvin.float_im_to_int(datum["image"]), "b c h w -> b h w c")
flow = einops.rearrange(datum["flow"], "b c h w -> b h w c")

cidx = 7
calvin.visualize_flow_vectors(images[cidx], flow[cidx], step=4, save_path=None, title=caption)
