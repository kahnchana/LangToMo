import av
import einops
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.models import optical_flow

from src.dataset import flow_utils

_IMAGE_SIZE = (512, 512)


def load_webm_with_pyav(video_path, verbose=False, as_array=False, resize_dims=None):
    """
    Loads a WebM video and reads its frames using PyAV, with optional resizing.

    Args:
        video_path (str): The path to the WebM video file.
        verbose (bool): If True, prints detailed information during loading.
        as_array (bool): If True, returns frames as a single NumPy array.
        resize_dims (tuple, optional): A tuple (width, height) to resize the frames to.
                                       If None, frames are loaded at their original resolution.
    """
    frame_list = []
    try:
        container = av.open(video_path)
        print(f"Loading video: {video_path}")

        # Iterate over video streams
        video_stream = next(s for s in container.streams if s.type == "video")
        if verbose:
            print(f"FPS: {video_stream.average_rate}")
            print(f"Original Resolution: {video_stream.width}x{video_stream.height}")

        # Decode frames
        for frame in container.decode(video_stream):
            # 'frame' is an av.VideoFrame object. Convert to NumPy array.
            if resize_dims:
                width, height = resize_dims
                img = frame.to_ndarray(format="rgb24", width=width, height=height)
            else:
                img = frame.to_ndarray(format="rgb24")

            frame_list.append(img)

            if verbose:
                print(f"Read frame at time {frame.time}, shape: {img.shape}")

        container.close()

        if as_array:
            frame_list = np.array(frame_list)

    except Exception as e:
        print(f"Error loading video with PyAV: {e}")
        frame_list = None

    return frame_list


def adaptive_normalize(flow, sf_x=20, sf_y=20, no_clamp=True):
    """
    Normalize the flow to the range [-1, 1]. Flow values higher than sf_x and sf_y are mapped beyond [-1,1] range.
    Args:
        flow: (B, C, H, W)
        sf_x: float
        sf_y: float
        no_clamp: bool
    Returns:
        flow_norm: (B, C, H, W)
    """
    assert flow.ndim == 4, "Set the shape of the flow input as (B, C, H, W)"
    assert sf_x is not None and sf_y is not None
    _, _, h, w = flow.shape

    max_clip_x = 1  # math.sqrt(w / sf_x) * 1.0
    max_clip_y = 1  # math.sqrt(h / sf_y) * 1.0

    flow_norm = flow.detach().clone()
    flow_x = flow[:, 0].detach().clone()
    flow_y = flow[:, 1].detach().clone()

    flow_x_norm = torch.sign(flow_x) * torch.sqrt(torch.abs(flow_x) / sf_x + 1e-7)
    flow_y_norm = torch.sign(flow_y) * torch.sqrt(torch.abs(flow_y) / sf_y + 1e-7)

    if no_clamp:
        flow_norm[:, 0] = flow_x_norm
        flow_norm[:, 1] = flow_y_norm
    else:
        flow_norm[:, 0] = torch.clamp(flow_x_norm, min=-max_clip_x, max=max_clip_x)
        flow_norm[:, 1] = torch.clamp(flow_y_norm, min=-max_clip_y, max=max_clip_y)

    return flow_norm


def adaptive_unnormalize(flow, sf_x=20, sf_y=20):
    # x: BCHW, optical flow
    assert flow.ndim == 4, "Set the shape of the flow input as (B, C, H, W)"
    assert sf_x is not None and sf_y is not None

    flow_orig = flow.detach().clone()
    flow_x = flow[:, 0].detach().clone()
    flow_y = flow[:, 1].detach().clone()

    flow_orig[:, 0] = torch.sign(flow_x) * sf_x * (flow_x**2 - 1e-7)
    flow_orig[:, 1] = torch.sign(flow_y) * sf_y * (flow_y**2 - 1e-7)

    return flow_orig


def generate_flow_online(
    image_pair, flow_model, flow_transform, normalize=(12, 8), third_channel=False
):
    flow_input, _ = flow_transform(image_pair, image_pair)
    start_im = flow_input[:-1]
    end_im = flow_input[1:]
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]

    flow_orig = flow_tensor

    if normalize is not None and not isinstance(normalize, bool):
        flow_tensor = adaptive_normalize(
            flow_tensor, sf_x=normalize[0], sf_y=normalize[1]
        )

    if third_channel:
        channel_0 = flow_tensor[:, 0:1, :, :]
        channel_1 = flow_tensor[:, 1:2, :, :]
        new_channel = (channel_0 + channel_1) / 2
        flow_tensor = torch.cat([channel_0, channel_1, new_channel], dim=1)

    return flow_orig, flow_tensor


def compare_pair(vector_a, vector_b):
    max_diff = torch.max(torch.abs(vector_a - vector_b)).item()
    mean_diff = torch.mean(torch.abs(vector_a - vector_b)).item()
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")


# Load the SDXL VAE & RAFT models.
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
flow_model = optical_flow.raft_large(
    weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False
).eval()
flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

# Load Video.
video_path = "/export/share/kranasinghe/dataset/ssv2/videos/78687.webm"
video = load_webm_with_pyav(video_path, resize_dims=(256, 256))
frame_pair = torch.stack([torch.from_numpy(x) for x in [video[14], video[16]]])
frame_pair = einops.rearrange(frame_pair, "b h w c -> b c h w")

# Generate Flow.
flow_vis_tensor, flow_vector = generate_flow_online(
    frame_pair, flow_model, flow_transform, normalize=(20, 20), third_channel=True
)
flow_vis = flow_utils.flow_to_pil_hsv(flow_vis_tensor[0], saturation=255, gamma=2.0)

# Move to device (e.g., GPU if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
img_tensor = flow_vector.to(device)

# Encode the image
with torch.no_grad():
    latent = vae.encode(img_tensor).latent_dist.sample()  # Sample latent code

# Decode the latent representation back to an image
with torch.no_grad():
    recon_img = vae.decode(latent).sample

# Compare Reconstruction.
print("\nReconstruction vs. Flow Vector")
compare_pair(flow_vector, recon_img.cpu())

# Post-process the reconstructed image to [0, 255] range
recon_img_unnorm = adaptive_unnormalize(recon_img[:, :2].cpu(), sf_x=20, sf_y=20)
recon_vis = flow_utils.flow_to_pil_hsv(recon_img_unnorm[0], saturation=255, gamma=2.0)

orig_vis = np.array(flow_vis)
recon_vis = np.array(recon_vis)

##################### View these Images #####################
vis_pair = Image.fromarray(np.concatenate([orig_vis, recon_vis], axis=1))
vis_diff = np.abs(orig_vis.astype(np.float32) - recon_vis.astype(np.float32))
vis_diff = Image.fromarray(255 - vis_diff.astype(np.uint8))
#############################################################

recon_loss = torch.abs(recon_img_unnorm - flow_vis_tensor).mean()
assert recon_loss < 0.1, f"Reconstruction L1 loss above 0.05: {recon_loss}"

print("\nReconstruction vs. Flow Vector in Unnormalized Space")
compare_pair(flow_vis_tensor, recon_img_unnorm)
