import os

import einops
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.models import optical_flow

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
    # Render the images as the gif:
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


def run_episode(all_env_dict, policy, cur_task, seed, save_root, resolution, camera="corner"):
    env = all_env_dict[cur_task](seed=seed)
    obs = env.reset()
    image = env.render(resolution=resolution, depth=False, camera_name="corner")
    image_vis = Image.fromarray(np.array(image))

    all_images_vis = [image_vis]
    all_actions = []
    run_episode = 2

    save_dir = f"{save_root}/{cur_task}_{seed:03d}"
    os.makedirs(save_dir, exist_ok=True)
    frame_counter = 0
    image_vis.save(f"{save_dir}/img_{frame_counter:03d}.png")
    frame_counter += 1

    while run_episode:
        a = policy.get_action(obs)
        obs, _, _, info = env.step(a)
        image = env.render(resolution=resolution, depth=False, camera_name=camera)
        image_vis = Image.fromarray(np.array(image))
        image_vis.save(f"{save_dir}/img_{frame_counter:03d}.png")
        frame_counter += 1
        all_images_vis.append(image_vis)
        all_actions.append(a)
        done = int(info["success"]) == 1
        if done:
            run_episode -= 1
        if run_episode != 2 and not done:
            break

    np.savez(f"{save_root}/{cur_task}_{seed:03d}.npz", actions=all_actions)

    return all_actions, all_images_vis


if __name__ == "__main__":
    all_env_dict = metaworld_env.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval().cuda()
    flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    RESOLUTION = (256, 256)
    SAVE_ROOT = "/home/kanchana/data/metaworld/mw_traj"

    mt50 = metaworld_env.MT50()

    task_list = [
        "door-open-v2-goal-observable",
        "door-close-v2-goal-observable",
        "basketball-v2-goal-observable",
        "shelf-place-v2-goal-observable",
        "button-press-v2-goal-observable",
        "button-press-topdown-v2-goal-observable",
        "faucet-close-v2-goal-observable",
        "faucet-open-v2-goal-observable",
        "handle-press-v2-goal-observable",
        "hammer-v2-goal-observable",
        "assembly-v2-goal-observable",
    ]

    for cur_task in task_list:
        print(f"Running {cur_task}")
        os.makedirs(f"{SAVE_ROOT}/{cur_task}", exist_ok=True)
        policy = task_to_policy[cur_task]()

        for seed in tqdm.tqdm(range(100)):
            if seed < 28:
                continue
            try:
                _ = run_episode(all_env_dict, policy, cur_task, seed, f"{SAVE_ROOT}/{cur_task}", RESOLUTION)
            except Exception:
                print(f"seed {seed} failed, skipping.")


TEST_RUN = False
if TEST_RUN:
    env = all_env_dict[cur_task](seed=seed)
    obs = env.reset()
    image = env.render(resolution=RESOLUTION, depth=False, camera_name="corner")
    image_vis = Image.fromarray(np.array(image))

    all_images = [image]
    all_images_vis = [image_vis]
    all_actions = []
    run_episode = 2

    save_dir = f"{SAVE_ROOT}/{cur_task}_{seed}"
    os.makedirs(save_dir, exist_ok=True)
    frame_counter = 0

    while run_episode:
        a = policy.get_action(obs)
        obs, _, _, info = env.step(a)
        image = env.render(resolution=RESOLUTION, depth=False, camera_name="corner")
        image_vis = Image.fromarray(np.array(image))
        image_vis.save(f"{save_dir}/img_{frame_counter:03d}.png")
        frame_counter += 1
        all_images.append(image)
        all_images_vis.append(image_vis)
        all_actions.append(a)
        done = int(info["success"]) == 1
        if done:
            run_episode -= 1
        if run_episode != 2 and not done:
            break

    np.savez(f"{SAVE_ROOT}/{cur_task}_{seed}.npz", actions=all_actions)

GEN_FLOW = False
if GEN_FLOW:
    all_images = torch.from_numpy(np.stack(all_images))
    N = len(all_images)
    generated_flows = []

    generated_flows = generate_flow_online(all_images[::8].cuda(), flow_model, flow_transform)
    generated_flows = einops.rearrange(generated_flows, "b c h w -> b h w c")
    normalizer = flow_utils.FlowNormalizer(*generated_flows.shape[1:3])
    unnorm_flows = generated_flows.cpu().numpy()

if False:
    from IPython import display

    vis_images = [
        flow_utils.visualize_flow_vectors_as_PIL(all_images[::8][i], unnorm_flows[i], step=8)
        for i in range(len(unnorm_flows))
    ]

    display.Image(as_gif([x.resize((512, 512)) for x in vis_images], duration=500))
