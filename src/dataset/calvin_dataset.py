import json
import os

import numpy as np
from torch.utils.data import Dataset


class RobotTrainingDataset(Dataset):
    def __init__(self, data_root, caption_file, transform=None, target_transform=None, frames_per_episode=8):
        """
        Args:
            data_root (str): Root directory where the data files are stored.
            caption_file (str): Path to the JSON file containing captions.
            transform (callable, optional): Transform to apply to the images.
            target_transform (callable, optional): Transform to apply to the flow targets.
            frames_per_episode (int): 8 by default
        """
        self.data_root = data_root
        self.caption_data = self._load_captions(caption_file)
        self.episode_ids = list(self.caption_data.keys())
        self.frames_per_episode = frames_per_episode
        self.sample_ids = list(range(len(self.episode_ids) * frames_per_episode))
        self.transform = transform
        self.target_transform = target_transform

    def _load_captions(self, caption_file):
        with open(caption_file, "r") as f:
            return json.load(f)

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
            flow = self.transform(flow)

        # Get the caption
        caption = self.caption_data[cur_name]

        return {"image": image, "flow": flow, "caption": caption}


# Example usage
if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")

    dataset = RobotTrainingDataset(DATA_ROOT, CAPTION_FILE)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(
            f"Sample {i}: Image shape: {sample['image'].shape}, "
            f"Flow shape: {sample['flow'].shape}, Caption: {sample['caption']}"
        )
        if i == 2:  # Display first 3 samples
            break
