import random
from glob import glob

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvideotransforms import video_transforms, volume_transforms


class MetaworldDataset(torch.utils.data.Dataset):
    def __init__(self, path, sample_per_seq=9, target_size=(128, 128), frameskip=None, randomcrop=True, captions=False):
        self.sample_per_seq = sample_per_seq
        self.frame_skip = frameskip
        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.embedding_dict = dict(np.load(f"{path}/use_embeddings.npz"))
        self.get_captions = captions
        for seq_dir in sequence_dirs:
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))

        if randomcrop:
            self.transform = video_transforms.Compose(
                [
                    video_transforms.CenterCrop((160, 160)),
                    video_transforms.RandomCrop((128, 128)),
                    video_transforms.Resize(target_size),
                    volume_transforms.ClipToTensor(),
                ]
            )
        else:
            self.transform = video_transforms.Compose(
                [
                    video_transforms.Resize(target_size),
                    volume_transforms.ClipToTensor(),
                ]
            )

    def get_samples(self, idx):
        seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        if self.frame_skip is None:
            start_idx = random.randint(0, len(seq) - 1)
            seq = seq[start_idx:]
            N = len(seq)
            samples = []
            for i in range(self.sample_per_seq - 1):
                samples.append(int(i * (N - 1) / (self.sample_per_seq - 1)))
            samples.append(N - 1)
        else:
            start_idx = random.randint(0, len(seq) - 1)
            samples = [
                i if i < len(seq) else -1
                for i in range(start_idx, start_idx + self.frame_skip * self.sample_per_seq, self.frame_skip)
            ]
        return [seq[i] for i in samples]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        images = self.transform([Image.open(s) for s in samples])  # [c f h w]
        x = rearrange(images, "c f h w -> f c h w")
        task = self.tasks[idx]
        task_embedding = self.embedding_dict[task]

        return_dic = {"observation": x, "caption_embedding": task_embedding}
        if self.get_captions:
            return_dic["caption"] = task

        return return_dic


class InfiniteWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            for i in range(len(self.dataset)):
                yield self.dataset[i]


if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/metaworld"
    dataset = MetaworldDataset(DATA_ROOT, captions=True)
    video_frames, text_embedding, text_caption = dataset[0]
    print(f"Target frames: {dataset[0]['observation'].shape}\nText command: {dataset[0]['caption_embedding'].shape}")
    print(f"Text caption: {dataset[0]['caption']}")

    dataset = MetaworldDataset(DATA_ROOT, captions=False)
    dataset = InfiniteWrapper(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True
    )
    for batch in dataloader:
        print(f"Target frames: {batch['observation'].shape}\nText command: {batch['caption_embedding'].shape}")
        break

    VIS_GIF = False
    if VIS_GIF:
        from IPython import display

        def as_gif(images, path="temp.gif", duration=100):
            # Render the images as the gif:
            images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
            return open(path, "rb").read()

        vis_trajectory = video_frames.transpose(0, 2, 3, 1)
        image_list = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]
        display.Image(as_gif([x.resize((512, 512)) for x in image_list], duration=200))

    CACHE_TEXT = False
    if CACHE_TEXT:
        import tensorflow_hub as hub

        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # Text descriptions for task labels.
        command_dict = {
            "assembly": "Assemble components by fitting them together.",
            "basketball": "Throw a ball into a basket.",
            "button press": "Press a button from the side.",
            "button press topdown": "Press a button from above.",
            "door close": "Close a door by pushing it shut.",
            "door open": "Open a door by pulling it open.",
            "drawer open": "Open a drawer by pulling it out.",
            "faucet close": "Turn a faucet handle to stop water flow.",
            "faucet open": "Turn a faucet handle to start water flow.",
            "hammer": "Use a hammer to drive a nail into a surface.",
            "handle press": "Press down a handle to operate a mechanism.",
            "push": "Move an object by applying force to slide it.",
            "shelf place": "Place an object onto a shelf.",
        }
        embedding_dict = {}
        for task, sentence in command_dict.items():
            embedding = model([sentence])
            embedding_dict[task] = embedding
        np.savez(f"{DATA_ROOT}/use_embeddings.npz", **embedding_dict)
