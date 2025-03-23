import random
from glob import glob

import numpy as np
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvideotransforms import video_transforms, volume_transforms


class SequentialDatasetv2(Dataset):
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
                    video_transforms.CenterCrop((128, 128)),
                    video_transforms.Resize(target_size),
                    volume_transforms.ClipToTensor(),
                ]
            )
        print("Done")

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
        x_cond = images[:, 0]  # first frame
        x = rearrange(images[:, 1:], "c f h w -> f c h w")  # all other frames
        task = self.tasks[idx]
        task_embedding = self.embedding_dict[task]

        if self.get_captions:
            return x, x_cond, task_embedding, task
        else:
            return x, x_cond, task_embedding


if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/metaworld"
    dataset = SequentialDatasetv2(DATA_ROOT, captions=True)
    target_frames, init_frame, text_embedding, text_caption = dataset[0]
    print(f"Target frames: {target_frames.shape}\nInit frame: {init_frame.shape}\nText command: {text_embedding.shape}")
    print(f"Text caption: {text_caption}")

    from IPython import display

    def as_gif(images, path="temp.gif", duration=100):
        # Render the images as the gif:
        images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        return open(path, "rb").read()

    vis_trajectory = np.concatenate([np.expand_dims(init_frame, axis=0), target_frames], axis=0).transpose(0, 2, 3, 1)
    image_list = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]
    display.Image(as_gif([x.resize((512, 512)) for x in image_list], duration=200))

    CACHE_TEXT = False

    if CACHE_TEXT:
        import tensorflow_hub as hub

        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # Text descriptions for task labels.
        command_dict = {
            "assembly": "Asse