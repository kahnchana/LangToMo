import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def as_gif(images, path="temp.gif", duration=100):
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


class FramePairDataset(Dataset):
    def __init__(self, root_dir, image_size=(128, 128), custom_transform=None, get_vis=False, get_caption=False):
        self.root_dir = root_dir
        self.transform = custom_transform or T.Compose(
            [
                T.Resize((image_size)),
                T.ToTensor(),
            ]
        )
        self.large_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )
        self.get_vis = get_vis
        self.get_caption = get_caption

        # Load captions
        caption_path = os.path.join(root_dir, "captions.json")
        self.captions = json.load(open(caption_path)) if os.path.exists(caption_path) else {}
        self.embedding_dict = dict(np.load(f"{root_dir}/use_embeddings.npz"))

        # Collect all valid video dirs
        self.video_dirs = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("vid_")]
        )

        # Build sample index: all possible t for each video
        self.samples = []

        for vid in self.video_dirs:
            st_dir = os.path.join(root_dir, f"{vid}/st")
            mid_dir = os.path.join(root_dir, f"{vid}/mid")
            end_dir = os.path.join(root_dir, f"{vid}/end")

            frames_st = sorted([f for f in os.listdir(st_dir) if f.endswith(".png")])
            frames_mid = sorted([f for f in os.listdir(mid_dir) if f.endswith(".png")])
            frames_end = sorted([f for f in os.listdir(end_dir) if f.endswith(".png")])

            # (start, mid) pairs
            for f1 in frames_st:
                for f2 in frames_mid:
                    self.samples.append((os.path.join(st_dir, f1), os.path.join(mid_dir, f2), "st2mid", vid))

            # (mid, end) pairs
            for f1 in frames_mid:
                for f2 in frames_end:
                    self.samples.append((os.path.join(mid_dir, f1), os.path.join(end_dir, f2), "mid2end", vid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fr1, fr2, kind, vid = self.samples[idx]
        frame_paths = [fr1, fr2]

        # Load and transform
        imgs_PIL = [Image.open(path).convert("RGB") for path in frame_paths]
        imgs = torch.stack([self.transform(x) for x in imgs_PIL], dim=0)
        imgs_256 = torch.stack([self.large_transform(x) for x in imgs_PIL], dim=0)

        caption = self.captions.get(vid, None)[kind]
        embedding = torch.from_numpy(self.embedding_dict[caption])

        return_dict = {
            "observation": imgs,
            "observation_256": imgs_256,
            "caption_embedding": embedding,
        }

        if self.get_vis:
            return_dict["vis"] = imgs_PIL
        if self.get_caption:
            return_dict["caption"] = caption
        return return_dict


if __name__ == "__main__":
    DATA_ROOT = "/nfs/ws2/kanchana/real_world/dataset_v2"
    DATA_ROOT = "/nfs/ws2/kanchana/real_world/dataset_v2_val"
    dataset = FramePairDataset(root_dir=DATA_ROOT, get_vis=True, get_caption=True)
    datum = dataset[0]

    train_dataset = FramePairDataset(root_dir=DATA_ROOT)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True
    )
    for batch in dataloader:
        print(f"Target frames: {batch['observation'].shape}\nText command: {batch['caption_embedding'].shape}")
        print(f"Large observation: {batch['observation_256'].shape}")
        break

    VIS = False
    if VIS:
        from IPython import display

        datum = dataset[11]
        display.Image(as_gif(datum["vis"], path="temp.gif", duration=500))

    CACHE_TEXT = False
    if CACHE_TEXT:
        import numpy as np
        import tensorflow_hub as hub

        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # Text descriptions for task labels.
        command_list = set([v for d in dataset.captions.values() for v in d.values()])
        embedding_dict = {}
        for sentence in command_list:
            embedding = model([sentence])
            embedding_dict[sentence] = embedding
        np.savez(f"{DATA_ROOT}/use_embeddings.npz", **embedding_dict)
