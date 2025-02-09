import json

import numpy as np
import torch
import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.dataset import flow_utils, ssv2


def get_optical_flow(vid_frames, model, transforms, device, samples=8):
    total_frames = vid_frames.shape[0]
    frame_subset_indices = np.linspace(0, total_frames - 1, samples + 1).astype(int).tolist()
    frame_subset = vid_frames[frame_subset_indices]
    frame_subset_tensor = float_im_to_int(frame_subset)
    frame_subset_tensor = torch.Tensor(frame_subset_tensor)

    raft_video, _ = transforms(frame_subset_tensor, frame_subset_tensor)
    start_frames = raft_video[:-1]
    end_frames = raft_video[1:]

    with torch.no_grad():
        flow_tensor = model(start_frames.to(device), end_frames.to(device), num_flow_updates=6)[-1]

    images = frame_subset[:samples]
    flow = flow_tensor.cpu()

    return images, flow


def float_im_to_int(frames):
    return (frames - frames.min()) / (frames.max() - frames.min())


if __name__ == "__main__":
    # Configs for CALVIN dataset, GPU device, and save path.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_grad_enabled(False)

    DATASET_ROOT = "/raid/datasets/ssv2"
    SAVE_ROOT = "/home/kanchana/data/ssv2_flow"
    dataset = ssv2.SSv2Dataset(
        json_path=f"{DATASET_ROOT}/annotations/something-something-v2-train.json",
        video_dir=f"{DATASET_ROOT}/videos/",
        num_frames=9,
        frame_dims=(256, 256),
    )

    # Optical Flow Generation.
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    raft_transforms = Raft_Large_Weights.DEFAULT.transforms()

    dataset_caption_info = {}
    normalizer = flow_utils.FlowNormalizer(200, 200)

    total_videos = len(dataset)
    for idx in tqdm.tqdm(range(total_videos)):
        video_id, vid_frames, caption = dataset[idx]
        images, flow = get_optical_flow(vid_frames, raft_model, raft_transforms, device)

        cur_name = f"vid_{video_id}"
        normalized_flow = (flow + 200) / (200 * 2)
        images = float_im_to_int(images)

        save_path = f"{SAVE_ROOT}/{cur_name}.npz"
        np.savez(save_path, flow=normalized_flow.numpy(), image=images[:8])
        dataset_caption_info[cur_name] = caption

    json.dump(dataset_caption_info, open(f"{SAVE_ROOT}/captions.json", "w"), indent=2)
