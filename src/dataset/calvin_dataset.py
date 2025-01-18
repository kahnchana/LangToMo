import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RobotTrainingDataset(Dataset):
    def __init__(
        self,
        data_root,
        caption_file,
        transform=None,
        target_transform=None,
        frames_per_episode=8,
        include_captions=False,
    ):
        """
        Args:
            data_root (str): Root directory where the data files are stored.
            caption_file (str): Path to the JSON file containing captions.
            transform (callable, optional): Transform to apply to the images.
            target_transform (callable, optional): Transform to apply to the flow targets.
            frames_per_episode (int): 8 by default
            include_captions (bool): Whether to include captions in the output.
        """
        self.data_root = data_root
        self.caption_data = self._load_captions(caption_file)
        self.caption_embeddings = self._load_caption_embeddings()
        self.episode_ids = list(self.caption_data.keys())
        self.frames_per_episode = frames_per_episode
        self.sample_ids = list(range(len(self.episode_ids) * frames_per_episode))
        self.transform = transform
        self.target_transform = target_transform
        self.include_captions = include_captions

    def _load_captions(self, caption_file):
        with open(caption_file, "r") as f:
            return json.load(f)

    def _load_caption_embeddings(self):
        embedding_file = os.path.join(self.data_root, "st5base_embeddings.npz")
        return dict(np.load(embedding_file))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Load a single data sample.

        Args:
            idx (int): Index of the episode to load.

        Returns:
            dict: Contains image, flow, and caption.
        """
        episode_id = self.episode_ids[idx // self.frames_per_episode]
        frame_id = idx % self.frames_per_episode

        cur_name = f"{episode_id}"
        datum_file = os.path.join(self.data_root, f"{cur_name}.npz")

        # Load the .npz file
        datum = np.load(datum_file)
        image = datum["image"][frame_id]
        flow = datum["flow"][frame_id]

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            flow = self.target_transform(flow)

        # Get the caption
        caption = self.caption_data[cur_name]
        caption_emb = self.caption_embeddings[cur_name].reshape(1, -1)

        if self.include_captions:
            return {"image": image, "flow": flow, "caption": caption, "caption_emb": caption_emb}
        else:
            return {"image": image, "flow": flow, "caption_emb": caption_emb}


class RobotVisualizationDataset(RobotTrainingDataset):
    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        """
        Load a single data sample.

        Args:
            idx (int): Index of the episode to load.

        Returns:
            dict: Contains image, flow, and caption.
        """
        episode_id = self.episode_ids[idx]

        cur_name = f"{episode_id}"
        datum_file = os.path.join(self.data_root, f"{cur_name}.npz")

        # Load the .npz file
        datum = np.load(datum_file)
        image = datum["image"]
        flow = datum["flow"]

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            flow = self.target_transform(flow)

        # Get the caption
        caption = self.caption_data[cur_name]
        caption_emb = self.caption_embeddings[cur_name].reshape(1, -1)

        return {"image": image, "flow": flow, "caption": caption, "caption_emb": caption_emb}


def get_transforms(image_size=(128, 128)):
    image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.from_numpy(x)),
            transforms.Resize(image_size),
        ]
    )

    flow_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.from_numpy(x)),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    )

    return image_transform, flow_transform


# Example usage
if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    dataset = RobotTrainingDataset(DATA_ROOT, CAPTION_FILE, include_captions=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(
            f"Sample {i}: Image shape: {sample['image'].shape}, "
            f"Flow shape: {sample['flow'].shape}, "
            f"Caption: {sample['caption']}, "
            f"Caption Emb: {sample['caption_emb'].shape}, "
        )
        if i == 2:  # Display first 3 samples
            break
