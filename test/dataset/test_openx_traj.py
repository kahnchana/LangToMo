import einops
import numpy as np
import tensorflow as tf
import torch
from IPython import display
from PIL import Image
from torchvision.models import optical_flow as of_utils

from src.dataset import flow_utils
from src.dataset import openx_trajectory_dataset as openx_traj


def as_gif(images, path="temp.gif", duration=100):
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


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


dataset_names = [
    "fractal20220817_data",
    "taco_play",
    "language_table",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "utaustin_mutex",
]
dataset_to_fps = {
    "fractal20220817_data": 3,
    "taco_play": 15,
    "language_table": 10,
    "stanford_hydra_dataset_converted_externally_to_rlds": 10,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": 3,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20,
    "utaustin_mutex": 20,
}
dataset_to_stride = {x: int(y // 3) for x, y in dataset_to_fps.items()}

traj_dataset = openx_traj.OpenXTrajectoryDataset(
    datasets=dataset_names,
    split="train[:10]",
    trajectory_length=9,
    traj_stride=dataset_to_stride,
    img_size=256,
    root_dir="/nfs/ws2/kanchana/openx",
)

mixed_dataset = tf.data.experimental.sample_from_datasets(
    list(traj_dataset.dataset_dict.values()), weights=[1 / len(dataset_names)] * len(dataset_names)
)

# Optical Flow setup.
flow_model = of_utils.raft_large(weights=of_utils.Raft_Large_Weights.DEFAULT, progress=False).eval().cuda()
flow_transform = of_utils.Raft_Large_Weights.DEFAULT.transforms()

# Inference
cur_ds = iter(traj_dataset.dataset_dict[dataset_names[0]])
for _ in range(10):
    trajectory = next(cur_ds)
frames = trajectory["observation"].numpy()
vis_trajectory = einops.rearrange(frames, "b c h w -> b h w c")
vis_images = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]

frames = torch.from_numpy(frames)
gen_flow = generate_flow_online(frames.cuda(), flow_model, flow_transform)


GET_ALL = False
if GET_ALL:
    ds_frame_dict = {dataset_names[0]: (vis_images, gen_flow.cpu())}
    for dataset_name in dataset_names[1:]:
        cur_ds = iter(traj_dataset.dataset_dict[dataset_name])
        for _ in range(10):
            trajectory = next(cur_ds)
            frames = trajectory["observation"].numpy()
        vis_trajectory = einops.rearrange(frames, "b c h w -> b h w c")
        vis_images = [Image.fromarray(x) for x in (vis_trajectory * 255).astype(np.uint8)]

        frames = torch.from_numpy(frames)
        gen_flow = generate_flow_online(frames.cuda(), flow_model, flow_transform)
        ds_frame_dict[dataset_name] = (vis_images, gen_flow.cpu())

GEN_VIS = False
if GEN_VIS:
    if GET_ALL:
        cur_name = dataset_names[0]
        vis_images, gen_flow = ds_frame_dict[cur_name]

    # Overlay as arrows
    vis_flow = einops.rearrange(gen_flow, "b c h w -> b h w c").cpu().numpy()
    overlay = [
        flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_flow[i], step=8) for i in range(len(gen_flow))
    ]
    display.Image(as_gif([x.resize((512, 512)) for x in overlay], duration=500))

    # Visualize flow as RGB image.
    flow_color = [flow_utils.flow_to_pil_hsv(x) for x in gen_flow]
    display.Image(as_gif([x.resize((512, 512)) for x in flow_color], duration=500))

    # RGG flow overlay.
    overlay_gif = [Image.blend(im, fl, alpha=0.5) for im, fl in zip(vis_images, flow_color)]
    display.Image(as_gif(overlay_gif, duration=500))
