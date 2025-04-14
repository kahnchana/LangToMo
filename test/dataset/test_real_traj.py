import glob

import diffusers
import einops
import imageio.v2 as imageio
import numpy as np
import tensorflow_hub as hub
import torch
import tqdm
from IPython import display
from PIL import Image
from torchvision.models import optical_flow as of_utils

from src import inference
from src.dataset import flow_utils


def as_gif(images, path="temp.gif", duration=100):
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


def as_mp4(images, path="temp.mp4", fps=10):
    imageio.mimsave(path, images, format="ffmpeg", fps=fps)
    with open(path, "rb") as f:
        video_bytes = f.read()
    return video_bytes


def generate_flow_online(image_condition, flow_model, flow_transform):
    flow_input, _ = flow_transform(image_condition, image_condition)
    start_im = flow_input[:-1]
    end_im = flow_input[1:]
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]
    return flow_tensor
    # flow_dim = flow_tensor.shape[-1]  # Image size is always square.
    # normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
    # return normalized_flow_tensor


class Dummy:
    pass


args = Dummy()
args.from_image = "/home/kanchana/repo/LangToMo/real_world/test_vid/frames/*.png"

args.model = "ox_ds7_001/model"
args.model_root = "/home/kanchana/repo/LangToMo/experiments"
args.text_emb = "/home/kanchana/data/metaworld/use_embeddings.npz"
args.text_emb = None

# Text embedding model.
lang_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Optical Flow setup.
flow_model = of_utils.raft_large(weights=of_utils.Raft_Large_Weights.DEFAULT, progress=False).eval().cuda()
flow_transform = of_utils.Raft_Large_Weights.DEFAULT.transforms()

# Diffusion Model
device = "cuda:0"
model_id = args.model
model_path = f"{args.model_root}/{model_id}"
model = diffusers.UNet2DConditionModel.from_pretrained(model_path, use_safetensors=True).to(device)

image_size = (512, 512)
task_embed = torch.from_numpy(lang_model(["Place orange duck on table."]).numpy())

image_list = sorted(glob.glob(args.from_image))
image_list = [Image.open(x) for x in image_list]
if image_list[0].mode != "RGB":
    image_list = [x.convert("RGB") for x in image_list]
image_list = [x.resize(image_size) for x in image_list]
image_list = [np.array(x) for x in image_list]
image_list = [x / 255.0 for x in image_list]
image_list = [x.transpose(2, 0, 1) for x in image_list]
image_list = [x[None, ...] for x in image_list]
image_list = [torch.from_numpy(x).to(torch.float32) for x in image_list]
frames = np.concatenate(image_list, axis=0)

vis_trajectory = einops.rearrange(frames[:-1], "b c h w -> b h w c")
vis_images = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]


image_size = (128, 128)
task_embed = torch.from_numpy(lang_model(["Place orange duck on table."]).numpy())

image_list = sorted(glob.glob(args.from_image))
image_list = [Image.open(x) for x in image_list]
if image_list[0].mode != "RGB":
    image_list = [x.convert("RGB") for x in image_list]
image_list = [x.resize(image_size) for x in image_list]
image_list = [np.array(x) for x in image_list]
image_list = [x / 255.0 for x in image_list]
image_list = [x.transpose(2, 0, 1) for x in image_list]
image_list = [x[None, ...] for x in image_list]
image_list = [torch.from_numpy(x).to(torch.float32) for x in image_list]
frames = np.concatenate(image_list, axis=0)

# Inference
frames = torch.from_numpy(frames).to(device)
gt_flow = generate_flow_online(frames, flow_model, flow_transform)
print(gt_flow.max(), gt_flow.min(), gt_flow.mean())

text_cond = task_embed.to(device).unsqueeze(0)
prev_flow = torch.ones_like(gt_flow[:1]).to(device) * 0.5
gen_flow = []
for fidx, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
    image_cond = frame.unsqueeze(0)
    image_cond = torch.cat([image_cond, prev_flow], dim=1)

    start_flow = torch.randn(prev_flow.shape, device=device)  # random noise
    generated_flow = inference.run_inference(model, start_flow, image_cond, text_cond, num_inference_steps=50)
    gen_flow.append(generated_flow)
    prev_flow = generated_flow
gen_flow = torch.cat(gen_flow)
vis_flow = gen_flow[:-1].cpu()
print(vis_flow.max(), vis_flow.min(), vis_flow.mean())


GEN_VIS = False
if GEN_VIS:
    # Images only.
    # display.Image(as_gif(vis_images, duration=500))
    display.Video(as_mp4([np.array(x) for x in vis_images], fps=2), embed=True)

    # Overlay as arrows
    vis_flow = torch.nn.functional.interpolate(vis_flow, size=(512, 512), mode="bilinear", align_corners=False)
    vis_flow = (vis_flow - 0.5) * 2 * 512

    vis_flow = einops.rearrange(vis_flow, "b c h w -> b h w c").cpu().numpy()
    overlay = [
        flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_flow[i], step=16) for i in range(len(vis_flow))
    ]
    # display.Image(as_gif([x.resize((512, 512)) for x in overlay], duration=500))
    display.Video(as_mp4([np.array(x) for x in overlay], fps=1), embed=True)

    # Visualize flow as RGB image.
    flow_color = [flow_utils.flow_to_pil_hsv(x) for x in vis_flow]
    # display.Image(as_gif([x.resize((512, 512)) for x in flow_color], duration=500))
    display.Video(as_mp4([np.array(x) for x in flow_color], fps=1), embed=True)

    # RGG flow overlay.
    overlay_gif = [Image.blend(im, fl, alpha=0.5) for im, fl in zip(vis_images, flow_color)]
    display.Video(as_mp4([np.array(x) for x in overlay_gif], fps=1), embed=True)
    # display.Image(as_gif(overlay_gif, duration=500))
