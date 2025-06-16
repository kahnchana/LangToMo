"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import diffusers
import einops
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision.models import optical_flow
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.dataset import flow_utils
from src.dataset import openx_trajectory_dataset as openx_traj

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--sub_datasets", type=str, nargs="+", default=["bridge"], help="List of sub-datasets")
    parser.add_argument("--val_sub_datasets", type=str, nargs="+", default=["bridge"], help="List of val sub-datasets")
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int, default=300_000, help="Total steps (overrides num_train_epochs)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="for debugging")
    parser.add_argument("--output_dir", type=str, default="ldm-test")
    parser.add_argument("--cache_dir", type=str, default=None, help="Dir for downloaded models / datasets.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=256, help="image resolution")
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Per device eval batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    # Scheduler options: linear/cosine/cosine_with_restarts/polynomial/constant/constant_with_warmup.
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=None)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--non_ema_revision", type=str, default=None, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--validation_steps", type=int, default=500, help="Val every every X steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Can set to `latest`")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


# def log_validation(dataloader, pipeline, args, accelerator, generator, flow_model, flow_transform):
#     RUN_COUNT = 25
#     count = 0
#     total_loss = 0
#     vis_index = np.random.choice(RUN_COUNT)
#     vis_pair = None
#     edited_images = []

#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     if torch.backends.mps.is_available():
#         autocast_ctx = nullcontext()
#     else:
#         autocast_ctx = torch.autocast(accelerator.device.type)

#     with autocast_ctx:
#         for idx, batch in tqdm(enumerate(dataloader), total=RUN_COUNT):
#             image_inputs = batch["observation"]
#             image_condition = image_inputs[:, 1]
#             clean_flow, prev_flow, vis_flow = generate_flow_online(
#                 args, batch["observation"], flow_model, flow_transform, normalize=(20, 20), third_channel=True
#             )
#             gen_flow = pipeline(
#                 args.validation_prompt,
#                 image=image_condition,
#                 num_inference_steps=20,
#                 image_guidance_scale=1.5,
#                 guidance_scale=7,
#                 generator=generator,
#             ).images[0]

#         for _ in range(args.num_validation_images):
#             edited_images.append(
#                 pipeline(
#                     args.validation_prompt,
#                     image=original_image,
#                     num_inference_steps=20,
#                     image_guidance_scale=1.5,
#                     guidance_scale=7,
#                     generator=generator,
#                 ).images[0]
#             )

#     for tracker in accelerator.trackers:
#         if tracker.name == "wandb":
#             wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
#             for edited_image in edited_images:
#                 wandb_table.add_data(wandb.Image(original_image), wandb.Image(edited_image), args.validation_prompt)
#             tracker.log({"validation": wandb_table})


def get_dataset(config, accelerator=None, tokenize_func=None):
    # Load dataset.
    if config.dataset_name == "openx":
        if config.sub_datasets[0] == "split_7_ds":
            traj_len = 3 if config.prev_flow else config.num_frames + 1
            data_root = "/home/kanchana/data/openx"
            dataset_names = list(openx_traj.DS_TO_FPS.keys())
            stride = 3
            dataset_to_stride = {x: int(y // stride) for x, y in openx_traj.DS_TO_FPS.items()}
            traj_dataset = openx_traj.OpenXTrajectoryDataset(
                root_dir=data_root,
                datasets=dataset_names,
                split="train",
                trajectory_length=traj_len,
                traj_stride=dataset_to_stride,
                img_size=config.resolution,
                infinite_repeat=True,
            )
            mixing_weights = openx_traj.get_ds_weights()
            mixed_dataset = tf.data.Dataset.sample_from_datasets(
                list(traj_dataset.dataset_dict.values()), weights=mixing_weights
            )

            val_ds_name = config.val_sub_datasets[0]
            val_dataset = openx_traj.OpenXTrajectoryDataset(
                root_dir=data_root,
                datasets=[val_ds_name],
                split="test",
                trajectory_length=traj_len,
                traj_stride=dataset_to_stride,
                img_size=config.resolution,
                infinite_repeat=True,
            )
            val_dataset = val_dataset.dataset_dict[val_ds_name]

            train_dataset = openx_traj.DatasetIterator(mixed_dataset, get_command=False)
            val_dataset = openx_traj.DatasetIterator(val_dataset, get_command=False)

        else:
            traj_len = 3
            sub_datasets = config.sub_datasets
            data_root = config.train_data_dir
            stride = 0.5
            dataset_to_stride = {x: int(y // stride) for x, y in openx_traj.DS_TO_FPS.items()}
            traj_dataset = openx_traj.OpenXTrajectoryDataset(
                root_dir=data_root,
                datasets=sub_datasets,
                split="train",
                trajectory_length=traj_len,
                traj_stride=dataset_to_stride,
                img_size=config.resolution,
                infinite_repeat=True,
            )
            dataset_name = sub_datasets[0]  # TODO: add support for multiple datasets
            train_dataset = openx_traj.DatasetIterator(
                traj_dataset.dataset_dict[dataset_name],
                get_command=True,
                get_language_embedding=False,
                get_obs_256=False,
                tokenize_func=tokenize_func,
            )

            val_name = config.val_sub_datasets[0]  # TODO: add support for multiple datasets
            val_dataset = openx_traj.OpenXTrajectoryDataset(
                root_dir=data_root,
                datasets=[val_name],
                split="train",
                trajectory_length=traj_len,
                traj_stride=dataset_to_stride,
                img_size=config.resolution,
            )
            val_dataset = openx_traj.DatasetIterator(
                val_dataset.dataset_dict[val_name],
                get_command=True,
                get_language_embedding=False,
                get_obs_256=False,
                tokenize_func=tokenize_func,
            )

        def collate_fn(examples):
            observation = torch.stack([example["observation"] for example in examples])
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"observation": observation, "input_ids": input_ids}

        # Ensure correct process management for sharded dataset.
        train_dataloader = accelerate.data_loader.IterableDatasetShard(
            train_dataset,
            batch_size=config.train_batch_size,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
        )
        val_dataloader = accelerate.data_loader.IterableDatasetShard(
            val_dataset,
            batch_size=config.eval_batch_size,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
        )
        # Still necessary for actual batching operations.
        train_dataloader = torch.utils.data.DataLoader(
            train_dataloader, batch_size=config.train_batch_size, num_workers=0, collate_fn=collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataloader, batch_size=config.eval_batch_size, num_workers=0, collate_fn=collate_fn
        )

    else:
        raise NotImplementedError(f"Dataset {config.dataset_name} not implemented")

    return train_dataloader, val_dataloader


def generate_flow_online(config, image_condition, flow_model, flow_transform, normalize=(12, 8), third_channel=True):
    flow_input, _ = flow_transform(image_condition, image_condition)
    start_im = einops.rearrange(flow_input[:, :-1], "b t c h w -> (b t) c h w")
    end_im = einops.rearrange(flow_input[:, 1:], "b t c h w -> (b t) c h w")
    with torch.no_grad():
        flow_tensor = flow_model(start_im, end_im, num_flow_updates=6)[-1]

    flow_orig = flow_tensor

    if normalize is not None and not isinstance(normalize, bool):
        flow_tensor = flow_utils.adaptive_normalize(flow_tensor, sf_x=normalize[0], sf_y=normalize[1])

    if third_channel:
        channel_0 = flow_tensor[:, 0:1, :, :]
        channel_1 = flow_tensor[:, 1:2, :, :]
        new_channel = (channel_0 + channel_1) / 2
        flow_tensor = torch.cat([channel_0, channel_1, new_channel], dim=1)

    flow_orig = einops.rearrange(flow_orig, "(b t) c h w -> b t c h w", b=image_condition.shape[0])
    flow_tensor = einops.rearrange(flow_tensor, "(b t) c h w -> b t c h w", b=image_condition.shape[0])

    assert flow_tensor.shape[1] == 2, "Flow tensor must have 2 time steps."
    clean_flow, prev_flow = flow_tensor[:, 1], flow_tensor[:, 0]

    return clean_flow, prev_flow, flow_orig


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Optical Flow Estimation Models.
    flow_model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
    flow_transform = optical_flow.Raft_Large_Weights.DEFAULT.transforms()

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    if not isinstance(model, diffusers.models.unets.unet_2d_condition.UNet2DConditionModel):
                        continue  # skip
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup Dataloader.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_dataloader, val_dataloader = get_dataset(args, accelerator=accelerator, tokenize_func=tokenize_captions)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    num_training_steps_for_scheduler = args.num_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler, flow_model, flow_transform = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler, flow_model, flow_transform
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.num_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            resume_step = resume_global_step

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.num_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    unet.train()
    train_loss = 0.0
    data_iter = iter(train_dataloader)

    while global_step < args.num_train_steps:
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and global_step < resume_step:
            if global_step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            continue

        # Handle infinite dataloader.
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch ended, restart iterator
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        except Exception as e:
            print(e)
            continue

        # Load data.
        image_inputs = batch["observation"]
        image_condition = image_inputs[:, 1]
        # text_condition = batch["input_ids"]
        # print(text_condition.shape)

        # Generate flow online.
        clean_flow, prev_flow, vis_flow = generate_flow_online(
            args, batch["observation"], flow_model, flow_transform, normalize=(20, 20), third_channel=True
        )

        with accelerator.accumulate(unet):
            # We want to learn the denoising process w.r.t the edited images which
            # are conditioned on the original image (which was edited) and the edit instruction.
            # So, first, convert images to latent space.
            latents = vae.encode(clean_flow.to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning.
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Get the additional image embedding for conditioning.
            # Instead of getting a diagonal Gaussian here, we simply take the mode.
            original_image_embeds = vae.encode(image_condition.to(weight_dtype)).latent_dist.mode()
            # prev_flow_embeds = vae.encode(prev_flow.to(weight_dtype)).latent_dist.mode()

            # Conditioning dropout to support classifier-free guidance during inference. For more details
            # check out the section 3.2.1 of the original paper https://huggingface.co/papers/2211.09800.
            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(bsz, device=latents.device, generator=generator)
                # Sample masks for the edit prompts.
                prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                # Final text conditioning.
                null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                # Sample masks for the original images.
                image_mask_dtype = original_image_embeds.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(bsz, 1, 1, 1)
                # Final image conditioning.
                original_image_embeds = image_mask * original_image_embeds

            # Concatenate the `original_image_embeds` with the `noisy_latents`.
            concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                ema_unet.step(unet.parameters())
            progress_bar.update(1)
            wandb_logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(wandb_logs, step=global_step)
            global_step += 1
            train_loss = 0.0

            # if global_step % args.validation_steps == 0:
            #     accelerator.wait_for_everyone()
            #     if accelerator.is_main_process:

            #         if args.use_ema:
            #             # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            #             ema_unet.store(unet.parameters())
            #             ema_unet.copy_to(unet.parameters())
            #         # The models need unwrapping because for compatibility in distributed training mode.
            #         pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            #             args.pretrained_model_name_or_path,
            #             unet=unwrap_model(unet),
            #             text_encoder=unwrap_model(text_encoder),
            #             vae=unwrap_model(vae),
            #             revision=args.revision,
            #             variant=args.variant,
            #             torch_dtype=weight_dtype,
            #         )

            #         log_validation(
            #             pipeline,
            #             args,
            #             accelerator,
            #             generator,
            #         )

            #         if args.use_ema:
            #             # Switch back to the original UNet parameters.
            #             ema_unet.restore(unet.parameters())

            #         del pipeline
            #         torch.cuda.empty_cache()

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.num_train_steps:
            break

    ##########################################################
    ############ Training is done, now save model ############
    ##########################################################

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # log_validation(
        #     pipeline,
        #     args,
        #     accelerator,
        #     generator,
        # )
    accelerator.end_training()


if __name__ == "__main__":
    main()
