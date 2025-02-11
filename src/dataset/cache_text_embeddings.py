import json

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer

SPLIT_OPTIONS = ["training", "validation"]
SPLIT = SPLIT_OPTIONS[1]
TASK_OPTIONS = ["D_D", "ABC_D"]
TASK = TASK_OPTIONS[1]


data_root = "/home/kanchana/data/ssv2_flow"
# train_data_root = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT_OPTIONS[0]}"
# val_data_root = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT_OPTIONS[1]}"

# DATA_ROOT = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT}"
DATA_ROOT = f"{data_root}/{SPLIT_OPTIONS[1]}"
caption_file = f"{DATA_ROOT}/captions.json"
caption_data = json.load(open(caption_file, "r"))

model = SentenceTransformer("sentence-transformers/sentence-t5-base")

emb_dict = {}

for key, caption in tqdm.tqdm(caption_data.items()):
    with torch.no_grad():
        emb_dict[key] = model.encode(caption)
    # break

np.savez(f"{DATA_ROOT}/st5base_embeddings.npz", **emb_dict)
