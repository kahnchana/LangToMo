import einops
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from src.dataset import flow_utils


def get_idx(dataset, info_dict, sel_idx=0):
    idx = info_dict["info"]["indx"][sel_idx]
    seq_length = idx[1] - idx[0]
    dataset.max_window_size, dataset.min_window_size = seq_length, seq_length
    start = dataset.episode_lookup.tolist().index(idx[0])
    seq_img = dataset[start]["rgb_obs"]["rgb_static"].numpy()
    lang_anno = info_dict["language"]["ann"][sel_idx]
    return seq_img, lang_anno


def float_im_to_int(img, factor=255, as_float=False):
    img = (img - img.min()) / (img.max() - img.min()) * factor
    if as_float:
        return img
    return img.astype(np.uint8)


def vis_image_grid(img_list, ncols=8):
    img_list = float_im_to_int(img_list)
    img_list = einops.rearrange(img_list, "(b1 b2) c h w -> (b1 h) (b2 w) c", b2=ncols)
    return Image.fromarray(img_list)


def save_images():
    save_imgs = float_im_to_int(frame_subset)
    save_imgs = einops.rearrange(save_imgs, "b c h w -> b h w c")
    save_imgs = [Image.fromarray(x) for x in save_imgs]

    for name, im in enumerate(save_imgs):
        im.save(f"im_{name:03d}.png")


def visualize_flow_vectors(image, flow, step=16, save_path=None, title="Optical Flow Vectors"):
    """
    Overlay optical flow vectors on an image.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3).
        flow (numpy.ndarray): Optical flow array (H, W, 2).
        step (int): Sampling step for displaying flow vectors.

    Returns:
        None (displays the visualization).
    """
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(np.int32)
    fx, fy = flow[x, y].T

    # Overlay flow vectors
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.quiver(x, y, fx, fy, color="red", angles="xy", scale_units="xy", scale=1, width=0.002)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Configs for CALVIN dataset and GPU device.
    ROOT_DIR = "/home/kanchana/repo/calvin"
    CONFIG_PATH = "../../../calvin/calvin_models/conf"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_grad_enabled(False)

    hydra.initialize(config_path=CONFIG_PATH, job_name="notebook_debug")

    overrides = [
        f"datamodule.root_data_dir={ROOT_DIR}/dataset/task_D_D/",
        "datamodule/observation_space=lang_rgb_static_rel_act",
    ]
    cfg = hydra.compose(config_name="lang_ann.yaml", overrides=overrides)

    # Loading CALVIN dataset.
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()
    dataset = data_module.train_dataloader()["vis"].dataset

    file_name = dataset.abs_datasets_dir / cfg.lang_folder / "auto_lang_ann.npy"
    ds_info = np.load(file_name, allow_pickle=True).item()

    vid_frames, caption = get_idx(dataset, ds_info, sel_idx=0)
    vis_img_all = vis_image_grid(vid_frames, ncols=8)
    vis_img_8 = vis_image_grid(vid_frames[::8,], ncols=8)

    # Optical Flow Generation.
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    transforms = Raft_Large_Weights.DEFAULT.transforms()

    total_frames = vid_frames.shape[0]
    frame_subset_indices = np.linspace(0, total_frames - 1, 9).astype(int).tolist()
    frame_subset = vid_frames[frame_subset_indices]
    frame_subset_tensor = float_im_to_int(frame_subset, factor=1, as_float=True)
    frame_subset_tensor = torch.Tensor(frame_subset_tensor)

    raft_video, _ = transforms(frame_subset_tensor, frame_subset_tensor)
    start_frames = raft_video[:-1]
    end_frames = raft_video[1:]

    flow_tensor = model(start_frames.to(device), end_frames.to(device), num_flow_updates=6)[-1]

    # Visualization.
    images = einops.rearrange(float_im_to_int(frame_subset), "b c h w -> b h w c")
    flow = einops.rearrange(flow_tensor, "b c h w -> b h w c").cpu()

    # Normalization
    normalizer = flow_utils.FlowNormalizer(flow.shape[1], flow.shape[2])
    normalized_flow = normalizer.normalize(flow)
    unnormalized_flow = normalizer.unnormalize(normalized_flow)
    assert np.allclose(flow, unnormalized_flow, atol=1e-4), "Unnormalized flow does not match the original!"
    # Visual Sanity Check.
    cidx = 7
    visualize_flow_vectors(images[cidx], flow[cidx], step=16, save_path=None)
    visualize_flow_vectors(images[cidx], unnormalized_flow[cidx], step=16, save_path=None)

    # Optional testing code.
    SAVE_VIS = False
    FORMAT_SAVE = False

    # Save visualization.
    if SAVE_VIS:
        for cidx in range(8):
            visualize_flow_vectors(images[cidx], flow[cidx], step=16, save_path=f"flow_{cidx:03d}.png", title=caption)
            visualize_flow_vectors(images[cidx], flow[cidx], step=16, save_path=None)

    # Test save formats.
    if FORMAT_SAVE:
        np.savez("joint_file.npz", flow=flow.numpy(), image=images[:8])

        np.save("flow_image.npy", flow[0].numpy())
        np.savez("flow_image.npz", flow[0].numpy())

        flow_image = (normalized_flow[0] * 65535).astype(np.uint16)  # Scale flow to uint16
        h, w, _ = flow_image.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint16)
        rgb_image[..., 0] = flow_image[..., 0]  # Map the horizontal flow to the red channel
        rgb_image[..., 1] = flow_image[..., 1]  # Map the vertical flow to the green channel
        Image.fromarray(rgb_image, mode="RGB").save("flow_image.png")
