import argparse
import glob
import os

import diffusers
import einops
import numpy as np
import tensorflow_hub as hub
import torch
import tqdm
from PIL import Image

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
    parser.add_argument("--model", type=str, default="mw_009")
    parser.add_argument("--model-root", type=str, default="/home/kanchana/repo/LangToMo/experiments")
    parser.add_argument("--data-root", type=str, default="/home/kanchana/data/calvin")
    parser.add_argument("--from-image", type=str, default=None)
    parser.add_argument("--text-emb", type=str, default=None)
    return parser.parse_args()


class Dummy:
    pass


if __name__ == "__main__":
    # args = parse_args()
    args = Dummy()
    args.model = "oxe_002"
    args.model_root = "/home/kanchana/repo/LangToMo/experiments"
    args.data_root = "/home/kanchana/data/calvin"
    args.from_image = "/home/kanchana/data/rlab_env/lab/*.png"
    args.text_emb = "/home/kanchana/data/metaworld/use_embeddings.npz"
    args.text_emb = None

    if args.text_emb is None:
        lang_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    SPLIT_OPTIONS = ["training", "validation"]
    TASK_OPTIONS = ["D_D", "ABC_D"]
    TASK = TASK_OPTIONS[1]

    DATA_ROOT = f"{args.data_root}/task_{TASK}/robot_{SPLIT_OPTIONS[1]}"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")
    device = "cuda:0"
    model_id = args.model
    model_path = f"{args.model_root}/{model_id}"

    # Load dataset.
    if args.from_image is None:
        image_size = (128, 128)
        _, flow_transform, val_image_transform = calvin_dataset.get_transforms(image_size)
        dataset = calvin_dataset.RobotVisualizationDataset(
            DATA_ROOT,
            CAPTION_FILE,
            transform=val_image_transform,
            target_transform=flow_transform,
            include_captions=True,
        )

        # Setup dataloader.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    else:
        image_size = (128, 128)
        task_name = os.path.basename(os.path.dirname(args.from_image))
        if args.text_emb is not None:
            embedding_dict = dict(np.load(args.text_emb))
            task_embed = torch.from_numpy(embedding_dict[" ".join(task_name.split("-")[:-3])])
        else:
            task_embed = torch.from_numpy(lang_model(["Pick up the orange duck."]).numpy())
        # image_list = sorted(glob.glob(args.from_image), key=lambda x: int(x[:-4].split("_")[-1]))
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
        image_list = [
            {"image": x, "flow": torch.ones_like(x[:, :2]) * 0.5, "caption_emb": task_embed} for x in image_list
        ]
        dataloader = image_list

    # Setup model.
    model = diffusers.UNet2DConditionModel.from_pretrained(model_path, use_safetensors=True).to(device)

    vis_count = 20
    save_dir = f"experiments/visualization/{model_id}"
    if args.from_image is not None:
        save_dir = os.path.dirname(args.from_image)
        vis_image_list = []
        vis_pred_list = []
        vis_count = len(image_list)
    os.makedirs(save_dir, exist_ok=True)
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=vis_count):
        if idx > vis_count and args.from_image is None:
            break

        if args.from_image is not None:
            clean_flow = batch["flow"].to(device)
            image_cond = batch["image"].to(device)
            image_cond = torch.cat([image_cond, clean_flow], dim=1)
            text_cond = batch["caption_emb"].to(device).unsqueeze(0)
            start_flow = torch.randn(clean_flow.shape, device=clean_flow.device)  # random noise
        else:
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

        if args.from_image is None:
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

            joint_video = [
                diffusers.utils.make_image_grid([x, y], rows=1, cols=2) for x, y in zip(gt_video, pred_video)
            ]

            output_path = f"{save_dir}/vis_{idx:04d}.gif"
            save_gif(joint_video, output_path)
        else:
            vis_image_list.append(vis_images[0])
            vis_pred_list.append(vis_pred[0])

    if args.from_image is not None:
        pred_video = [
            flow_utils.visualize_flow_vectors_as_PIL(vis_image_list[i][:, :, :3], vis_pred_list[i], step=4)
            for i in range(len(vis_image_list))
        ]
        # output_path = f"{save_dir}/vis_images.gif"
        # save_gif(pred_video, output_path)
