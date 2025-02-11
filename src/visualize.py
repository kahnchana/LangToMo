import argparse
import os

import diffusers
import einops
import numpy as np
import torch
import tqdm

from src import inference
from src.dataset import calvin_dataset, flow_utils


def save_gif(image_list, save_path, duration=200, loop=0, end_pause=3):
    extended_list = image_list + [image_list[-1]] * end_pause
    extended_list[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=extended_list[1:],  # Add the rest of the images
        duration=200,  # Duration between frames in milliseconds
        loop=0,  # Loop forever
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="test_006")
    parser.add_argument("--model-root", type=str, default="/home/kanchana/repo/LangToMo/experiments")
    parser.add_argument("--data-root", type=str, default="/home/kanchana/data/calvin")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SPLIT_OPTIONS = ["training", "validation"]
    TASK_OPTIONS = ["D_D", "ABC_D"]
    TASK = TASK_OPTIONS[1]

    DATA_ROOT = f"{args.data_root}/task_{TASK}/robot_{SPLIT_OPTIONS[1]}"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")
    device = "cuda:0"
    model_id = args.model
    model_path = f"{args.model_root}/{model_id}"

    # Load dataset.
    image_size = (128, 128)
    _, flow_transform, val_image_transform = calvin_dataset.get_transforms(image_size)
    dataset = calvin_dataset.RobotVisualizationDataset(
        DATA_ROOT, CAPTION_FILE, transform=val_image_transform, target_transform=flow_transform, include_captions=True
    )

    # Setup dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Setup model.
    model = diffusers.UNet2DConditionModel.from_pretrained(model_path, use_safetensors=True).to(device)

    vis_count = 20
    save_dir = f"experiments/visualization/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=vis_count):
        if idx > vis_count:
            break

        # Generate flow from pretrained model.
        clean_flow = batch["flow"].to(device)[0]
        image_cond = batch["image"].to(device)[0]
        frame_count = clean_flow.shape[0]
        text_cond = batch["caption_emb"].to(device).repeat(frame_count, 1, 1)
        start_flow = torch.randn(clean_flow.shape, device=clean_flow.device)  # random noise

        generated_flow = inference.run_inference(model, start_flow, image_cond, text_cond, num_inference_steps=50)

        # Generate visualiation.
        vis_images = einops.rearrange((image_cond.cpu().numpy() * 255).astype(np.uint8), "b c h w -> b h w c")
        normalizer = flow_utils.FlowNormalizer(*image_size)
        vis_pred = einops.rearrange(generated_flow.cpu().numpy(), "b c h w -> b h w c")
        vis_pred = normalizer.unnormalize(vis_pred)
        vis_gt = einops.rearrange(clean_flow.cpu().numpy(), "b c h w -> b h w c")
        vis_gt = normalizer.unnormalize(vis_gt)

        caption = batch["caption"]
        gt_video = [
            flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_gt[i], step=4, title=f"GT: {caption}")
            for i in range(frame_count)
        ]
        pred_video = [
            flow_utils.visualize_flow_vectors_as_PIL(vis_images[i], vis_pred[i], step=4, title=f"Pred: {caption}")
            for i in range(frame_count)
        ]

        joint_video = [diffusers.utils.make_image_grid([x, y], rows=1, cols=2) for x, y in zip(gt_video, pred_video)]

        output_path = f"{save_dir}/vis_{idx:04d}.gif"
        save_gif(joint_video, output_path)
