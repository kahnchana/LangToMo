import diffusers
import einops
import numpy as np
import torch
from IPython import display
from PIL import Image
from torchvision.models import optical_flow as of_utils

import metaworld as metaworld_env
from metaworld.policies import (
    sawyer_assembly_v2_policy,
    sawyer_basketball_v2_policy,
    sawyer_button_press_topdown_v2_policy,
    sawyer_button_press_v2_policy,
    sawyer_door_close_v2_policy,
    sawyer_door_open_v2_policy,
    sawyer_faucet_close_v2_policy,
    sawyer_faucet_open_v2_policy,
    sawyer_hammer_v2_policy,
    sawyer_handle_press_v2_policy,
    sawyer_shelf_place_v2_policy,
)
from src.dataset import flow_utils


def as_gif(images, path="temp.gif", duration=100):
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


def generate_flow_online(image_condition, flow_model, flow_transform):
    image_condition = einops.rearrange(image_condition, "b h w c -> b c h w")
    flow_input, _ = flow_transform(image_condition, image_condition)
    start_im = flow_input[:-1]
    end_im = flow_input[1:]
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]
    return flow_tensor
    # flow_dim = flow_tensor.shape[-1]  # Image size is always square.
    # normalized_flow_tensor = (flow_tensor + flow_dim) / (flow_dim * 2)
    # return normalized_flow_tensor


task_to_policy = {
    "door-open-v2-goal-observable": sawyer_door_open_v2_policy.SawyerDoorOpenV2Policy,
    "door-close-v2-goal-observable": sawyer_door_close_v2_policy.SawyerDoorCloseV2Policy,
    "basketball-v2-goal-observable": sawyer_basketball_v2_policy.SawyerBasketballV2Policy,
    "shelf-place-v2-goal-observable": sawyer_shelf_place_v2_policy.SawyerShelfPlaceV2Policy,
    "button-press-v2-goal-observable": sawyer_button_press_v2_policy.SawyerButtonPressV2Policy,
    "button-press-topdown-v2-goal-observable": sawyer_button_press_topdown_v2_policy.SawyerButtonPressTopdownV2Policy,
    "faucet-close-v2-goal-observable": sawyer_faucet_close_v2_policy.SawyerFaucetCloseV2Policy,
    "faucet-open-v2-goal-observable": sawyer_faucet_open_v2_policy.SawyerFaucetOpenV2Policy,
    "handle-press-v2-goal-observable": sawyer_handle_press_v2_policy.SawyerHandlePressV2Policy,
    "hammer-v2-goal-observable": sawyer_hammer_v2_policy.SawyerHammerV2Policy,
    "assembly-v2-goal-observable": sawyer_assembly_v2_policy.SawyerAssemblyV2Policy,
}

all_env_dict = metaworld_env.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
RESOLUTION = (256, 256)

mt50 = metaworld_env.MT50()
cur_task = "door-open-v2-goal-observable"
policy = task_to_policy[cur_task]()

seed = 7
env = all_env_dict[cur_task](seed=seed)

obs = env.reset()
image = env.render(resolution=RESOLUTION, depth=False, camera_name="corner")
image_vis = Image.fromarray(np.array(image))


# Optical Flow setup.
flow_model = of_utils.raft_large(weights=of_utils.Raft_Large_Weights.DEFAULT, progress=False).eval().cuda()
flow_transform = of_utils.Raft_Large_Weights.DEFAULT.transforms()

# Model setup.
device = "cuda:0"
model_id = "mw_010/model"
model_path = f"/home/kanchana/repo/LangToMo/experiments/{model_id}"
model = diffusers.UNet2DConditionModel.from_pretrained(model_path, use_safetensors=True).to(device)

all_images = [image]
all_images_vis = [image_vis]
all_actions = []
run_episode = 2
while run_episode:
    a = policy.get_action(obs)
    obs, _, _, info = env.step(a)
    image = env.render(resolution=RESOLUTION, depth=False, camera_name="corner")
    image_vis = Image.fromarray(np.array(image))
    all_images.append(image)
    all_images_vis.append(image_vis)
    all_actions.append(a)
    done = int(info["success"]) == 1
    if done:
        run_episode -= 1
    if run_episode != 2 and not done:
        break

all_images = torch.from_numpy(np.stack(all_images))
selected_images = all_images[::8]
generated_flows = generate_flow_online(selected_images.to(device), flow_model, flow_transform)


GEN_VIS = False
if GEN_VIS:
    # Images only.
    vis_images = [Image.fromarray(x) for x in selected_images[:-1].numpy()]
    display.Image(as_gif(vis_images, duration=500))

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
