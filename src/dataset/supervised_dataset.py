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


DATASET_LIST = [
    "door-open-v2-goal-observable",
    "door-close-v2-goal-observable",
    "basketball-v2-goal-observable",
    "shelf-place-v2-goal-observable",
    "button-press-v2-goal-observable",
    "button-press-topdown-v2-goal-observable",
    "faucet-close-v2-goal-observable",
    "faucet-open-v2-goal-observable",
    "handle-press-v2-goal-observable",
    "hammer-v2-goal-observable",
    "assembly-v2-goal-observable",
]


class SupervisedDataset(Dataset):
    def __init__(self, root_dir_list, k=10, image_size=(128, 128), custom_transform=None, get_vis=False):
        if isinstance(root_dir_list, str):
            root_dir_list = [root_dir_list]
        self.root_dir_list = root_dir_list
        self.k = k
        self.transform = custom_transform or T.Compose(
            [
                T.Resize(image_size),
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

        # Collect all valid video dirs
        self.video_dirs = []
        for root in root_dir_list:
            for d in os.listdir(root):
                full_path = os.path.join(root, d)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(root, f"{d}.npz")):
                    self.video_dirs.append((root, d))

        # Build index of frames
        self.video_frames = {}
        for root, vid in self.video_dirs:
            vid_path = os.path.join(root, vid)
            frames = sorted([f for f in os.listdir(vid_path) if f.endswith(".png")])
            if len(frames) > 0:
                self.video_frames[(root, vid)] = frames

        # Build sample index: all possible t for each video
        self.samples = []
        for (root, vid), frames in self.video_frames.items():
            for t in range(len(frames)):
                self.samples.append((root, vid, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        root, vid, t = self.samples[idx]
        frames = self.video_frames[(root, vid)]
        num_frames = len(frames)
        frame_dir = os.path.join(root, vid)

        # Load actions
        actions_path = os.path.join(root, f"{vid}.npz")
        actions = np.load(actions_path)["actions"].astype(np.float32)
        actions = np.concatenate([actions, np.zeros((1, actions.shape[1]), dtype=np.float32)])

        # Compute t + k
        idx_tp_k = min(t + self.k, num_frames - 1)

        # Random i in [0, k), ensure t+i < num_frames
        valid_i_vals = [i for i in range(0, self.k) if t + i < num_frames]
        if not valid_i_vals:
            idx_tp_i = t
        else:
            i = np.random.choice(valid_i_vals)
            idx_tp_i = t + i

        # Frame paths
        frame_paths = [
            os.path.join(frame_dir, frames[t]),  # t
            os.path.join(frame_dir, frames[idx_tp_i]),  # t+i
            os.path.join(frame_dir, frames[idx_tp_k]),  # t+k
        ]

        # Load and transform frames
        imgs_PIL = [Image.open(path).convert("RGB") for path in frame_paths]
        imgs = torch.stack([self.transform(x) for x in imgs_PIL], dim=0)
        imgs_256 = torch.stack([self.large_transform(x) for x in imgs_PIL], dim=0)

        # Load action for t+i
        action_tp_i = torch.from_numpy(actions[idx_tp_i])

        return_dict = {
            "observation": imgs,  # [t, t+k, t+i]
            "observation_256": imgs_256,
            "action": action_tp_i,
        }

        if self.get_vis:
            return_dict["vis"] = imgs_PIL

        return return_dict


if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/metaworld/mw_traj"
    # dataset_dir = f"{DATA_ROOT}/door-open-v2-goal-observable"
    dataset_dir = [f"{DATA_ROOT}/{d}" for d in DATASET_LIST]
    dataset = SupervisedDataset(root_dir_list=dataset_dir, k=10, get_vis=True)
    print(f"Dataset length: {len(dataset)}")
    datum = dataset[0]

    train_dataset = SupervisedDataset(root_dir_list=dataset_dir, k=10)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True
    )

    for batch in dataloader:
        print(f"Target frames: {batch['observation'].shape}")
        print(f"Large observation: {batch['observation_256'].shape}")
        print(f"Action: {batch['action'].shape}")
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
        command_list = set(list(dataset.captions.values()))
        embedding_dict = {}
        for sentence in command_list:
            embedding = model([sentence])
            embedding_dict[sentence] = embedding
        np.savez(f"{DATA_ROOT}/use_embeddings.npz", **embedding_dict)
