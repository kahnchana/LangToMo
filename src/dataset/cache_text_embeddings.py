import json

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer

SPLIT = "train"

if SPLIT == "train":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
else:
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_validation"
caption_file = f"{DATA_ROOT}/captions.json"
caption_data = json.load(open(caption_file, "r"))

model = SentenceTransformer("sentence-transformers/sentence-t5-base")

emb_dict = {}

for key, caption in tqdm.tqdm(caption_data.items()):
    with torch.no_grad():
        emb_dict[key] = model.encode(caption)
    # break

np.savez(f"{DATA_ROOT}/st5base_embeddings.npz", **emb_dict)
