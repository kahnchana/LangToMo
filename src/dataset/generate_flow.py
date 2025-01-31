import json

import hydra
import numpy as np
import torch
import tqdm
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.dataset import calvin, flow_utils


def get_optical_flow(vid_frames, model, transforms, device, samples=8):
    total_frames = vid_frames.shape[0]
    frame_subset_indices = np.linspace(0, total_frames - 1, samples + 1).astype(int).tolist()
    frame_subset = vid_frames[frame_subset_indices]
    frame_subset_tensor = calvin.float_im_to_int(frame_subset, factor=1, as_float=True)
    frame_subset_tensor = torch.Tensor(frame_subset_tensor)

    raft_video, _ = transforms(frame_subset_tensor, frame_subset_tensor)
    start_frames = raft_video[:-1]
    end_frames = raft_video[1:]

    with torch.no_grad():
        flow_tensor = model(start_frames.to(device), end_frames.to(device), num_flow_updates=6)[-1]

    images = frame_subset[:samples]
    flow = flow_tensor.cpu()

    return images, flow


if __name__ == "__main__":
    # Configs for CALVIN dataset, GPU device, and save path.
    ROOT_DIR = "/home/kanchana/repo/calvin"
    CONFIG_PATH = "../../../calvin/calvin_models/conf"

    SPLIT_OPTIONS = ["training", "validation"]
    SPLIT = SPLIT_OPTIONS[0]
    TASK_OPTIONS = ["D_D", "ABC_D"]
    TASK = TASK_OPTIONS[1]

    SAVE_ROOT = f"/home/kanchana/data/calvin/task_{TASK}/robot_{SPLIT}"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_grad_enabled(False)

    hydra.initialize(config_path=CONFIG_PATH, job_name="notebook_debug")

    overrides = [
        f"datamodule.root_data_dir={ROOT_DIR}/dataset/task_{TASK}/",
        "datamodule/observation_space=lang_rgb_static_rel_act",
    ]
    cfg = hydra.compose(config_name="lang_ann.yaml", overrides=overrides)

    # Loading CALVIN dataset.
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    if SPLIT == "training":
        dataset = data_module.train_dataloader()["vis"].dataset
    else:
        dataset = data_module.val_dataloader().dataset.datasets["vis"]

    file_name = dataset.abs_datasets_dir / cfg.lang_folder / "auto_lang_ann.npy"
    ds_info = np.load(file_name, allow_pickle=True).item()
    total_episodes = len(ds_info["language"]["ann"])

    # Optical Flow Generation.
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    raft_transforms = Raft_Large_Weights.DEFAULT.transforms()

    dataset_caption_info = {}
    normalizer = flow_utils.FlowNormalizer(200, 200)

    for episode_idx in tqdm.tqdm(range(total_episodes)):
        vid_frames, caption = calvin.get_idx(dataset, ds_info, sel_idx=episode_idx)
        images, flow = get_optical_flow(vid_frames, raft_model, raft_transforms, device)

        cur_name = f"eps_{episode_idx:05d}"
        normalized_flow = (flow + 200) / (200 * 2)
        images = calvin.float_im_to_int(images, 1, True)

        save_path = f"{SAVE_ROOT}/{cur_name}.npz"
        np.savez(save_path, flow=normalized_flow.numpy(), image=images[:8])
        dataset_caption_info[cur_name] = caption

    json.dump(dataset_caption_info, open(f"{SAVE_ROOT}/captions.json", "w"), indent=2)
