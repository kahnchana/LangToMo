import json
import math
import os
import random
from typing import Dict, Tuple

import av
import einops
import numpy as np
import torch
from PIL import Image


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def pyav_decode_stream(container, start_pts, end_pts, stream, stream_name, buffer_size=0):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips=10,
    target_fps=30,
    start=None,
    end=None,
    duration=None,
    frames_length=None,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)

    orig_duration = duration
    tb = float(container.streams.video[0].time_base)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration
    if duration is None and orig_duration is not None:
        duration = orig_duration / tb

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    if start is not None and end is not None:
        decode_all_video = False

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        if start is None and end is None:
            video_frames, _ = pyav_decode_stream(
                container,
                video_start_pts,
                video_end_pts,
                container.streams.video[0],
                {"video": 0},
            )
        else:
            timebase = duration / frames_length
            start_i = start
            end_i = end
            video_frames, _ = pyav_decode_stream(
                container,
                start_i,
                end_i,
                container.streams.video[0],
                {"video": 0},
            )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))

    return frames, fps, decode_all_video


class SSv2Dataset:
    def __init__(self, json_path: str, video_dir: str, num_frames: int = 8, frame_dims=None):
        """
        Initialize the SSv2 dataset loader using Decord.

        Args:
            json_path (str): Path to the JSON file containing video ID to label mappings.
            video_dir (str): Path to the directory containing WebM videos.
            num_frames (int): Number of frames to load for each video.
        """
        self.json_path = json_path
        self.video_dir = video_dir
        self.data = self._load_json()
        self.num_frames = num_frames
        self.frame_dims = frame_dims

    def _load_json(self) -> Dict[str, str]:
        """Load video ID to label mapping from JSON file."""
        with open(self.json_path, "r") as f:
            data = json.load(f)
        return data  # {video_id: label}

    def get_video_path(self, video_id: str) -> str:
        """Return the full path to a video file given its ID."""
        return os.path.join(self.video_dir, f"{video_id}.webm")

    def resize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            frames,
            size=self.frame_dims,
            mode="bilinear",
            align_corners=False,
        )

    def load_video_frames(self, video_id: str) -> np.ndarray:
        """
        Load video frames using PyAV.

        Args:
            video_id (str): Video ID.

        Returns:
            Numpy array of shape (num_frames, H, W, 3).
        """
        video_path = self.get_video_path(video_id)
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found.")
            return np.array([])

        try:
            container = av.open(video_path)
            target_fps = 30  # borrow from Kinetics code

            frames, _, _ = pyav_decode(  # frames, fps, decode_all_video
                container=container,
                sampling_rate=1,
                num_frames=self.num_frames,
                clip_idx=0,  # clip idx
                num_clips=1,  # num clips
                target_fps=target_fps,
            )

            # clip_sz = self.num_frames / target_fps * fps
            # start_idx, end_idx = get_start_end_idx(
            #     video_size=frames.shape[0],
            #     clip_size=clip_sz,
            #     clip_idx=0,
            #     num_clips=1,
            # )
            start_idx = 0
            end_idx = frames.shape[0] - 1
            frames = temporal_sampling(frames, start_idx, end_idx, self.num_frames)  # frames.shape = (T, H, W, C)
            frames = einops.rearrange(frames, "t h w c -> t c h w")

            if self.frame_dims is not None:
                frames = self.resize_frames(frames)

            return frames

        except Exception as e:
            print(f"Error loading video {video_id}: {e}")
            return np.array([])

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, str]:
        """
        Get video frames and label by index.

        Args:
            index (int): Index of the video in the dataset.

        Returns:
            Tuple of (video ID, frames (numpy array), label).
        """
        current_datum = self.data[index]
        video_id = current_datum["id"]
        caption = current_datum["label"].capitalize()
        frames = self.load_video_frames(video_id)

        return video_id, frames, caption

    @staticmethod
    def vis_frames(frames, rows=1):
        vis = einops.rearrange(frames, "(t1 t2) c h w -> (t1 h) (t2 w) c", t1=rows)
        return Image.fromarray(np.array(vis))


# Example Usage
if __name__ == "__main__":
    DATASET_ROOT = "/raid/datasets/ssv2"
    dataset = SSv2Dataset(
        json_path=f"{DATASET_ROOT}/annotations/something-something-v2-train.json",
        video_dir=f"{DATASET_ROOT}/videos/",
        num_frames=8,
        frame_dims=(256, 256),
    )

    # Example: Get one video.
    video_id, frames, label = dataset[0]
    print(f"Video ID: {video_id}, Label: {label}, Frames Shape: {frames.shape if frames.size else 'None'}")

    # Visualize example.
    idx = 85
    dataset.num_frames = 8
    video_id, frames, label = dataset[idx]
    print(label)
    print(frames.shape)
    # dataset.vis_frames(frames, 2)
