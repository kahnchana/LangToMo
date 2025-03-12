import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RobotTrainingDataset(Dataset):
    def __init__(
        self,
        data_root,
        caption_file,
        action_file=None,
        transform=None,
        target_transform=None,
        frames_per_episode=8,
        include_captions=False,
        include_actions=False,
        include_prev_flow: bool = False,
    ):
        """
        Args:
            data_root (str): Root directory where the data files are stored.
            caption_file (str): Path to the JSON file containing captions.
            action_file (str, optional): Path to the JSON file containing actions.
            transform (callable, optional): Transform to apply to the images.
            target_transform (callable, optional): Transform to apply to the flow targets.
            frames_per_episode (int): 8 by default
            include_captions (bool): Whether to include captions in the output.
            include_actions (bool): Whether to include actions in the output.
            include_prev_flow (bool): Whether to include previous flow in the output.
        """
        self.data_root = data_root
        self.caption_data = self._load_json(caption_file)
        self.caption_embeddings = self._load_caption_embeddings()
        self.episode_ids = list(self.caption_data.keys())
        self.frames_per_episode = frames_per_episode
        self.sample_ids = list(range(len(self.episode_ids) * frames_per_episode))
        self.transform = transform
        self.target_transform = target_transform
        self.include_captions = include_captions
        self.include_actions = include_actions
        self.action_data = self._load_actions(action_file)
        self.include_prev_flow = include_prev_flow
        self.prev_flow_random = 0.5

    def _load_json(self, json_file):
        with open(json_file, "r") as f:
            return json.load(f)

    def _load_caption_embeddings(self):
        embedding_file = os.path.join(self.data_root, "st5base_embeddings.npz")
        return dict(np.load(embedding_file))

    def _load_actions(self, action_file):
        if self.include_actions:
            assert action_file is not None
            action_data = self._load_json(action_file)
        else:
            action_data = None
        return action_data

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Load a single data sample.

        Args:
            idx (int): Index of the episode to load.

        Returns:
            dict: Contains image, flow, and caption.
        """
        episode_id = self.episode_ids[idx // self.frames_per_episode]
        frame_id = idx % self.frames_per_episode

        cur_name = f"{episode_id}"
        datum_file = os.path.join(self.data_root, f"{cur_name}.npz")

        # Load the .npz file
        datum = np.load(datum_file)
        image = datum["image"][frame_id]
        flow = datum["flow"][frame_id]

        output_seq = [image, flow]
        if self.include_prev_flow:
            prev_flow = datum["flow"][min(0, frame_id - 1)]
            pick = random.random() < self.prev_flow_random
            if pick and frame_id > 0:
                prev_flow = datum["flow"][frame_id - 1]
            else:
                prev_flow = np.zeros_like(flow)
            output_seq.append(prev_flow)

        # Apply transformations if any
        if self.transform is not None:
            output_seq = self.transform(tuple(output_seq))

        if self.target_transform is not None:  # to be depracated
            flow = self.target_transform(flow)

        # Get the caption
        caption = self.caption_data[cur_name]
        caption_emb = self.caption_embeddings[cur_name].reshape(1, -1)

        # Get the action
        if self.include_actions:
            action = torch.from_numpy(np.array(self.action_data[cur_name][frame_id], dtype=np.float32))
        else:
            action = None

        return_dict = {"image": output_seq[0], "flow": output_seq[1], "caption_emb": caption_emb}

        if self.include_captions:
            return_dict["caption"] = caption
        if self.include_actions:
            return_dict["relative_action"] = action
        if self.include_prev_flow:
            assert len(output_seq) == 3, f"No prev_flow in output sequence of length {len(output_seq)}"
            return_dict["prev_flow"] = output_seq[2]

        return return_dict


class RobotVisualizationDataset(RobotTrainingDataset):
    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        """
        Load a single data sample.

        Args:
            idx (int): Index of the episode to load.

        Returns:
            dict: Contains image, flow, and caption.
        """
        episode_id = self.episode_ids[idx]

        cur_name = f"{episode_id}"
        datum_file = os.path.join(self.data_root, f"{cur_name}.npz")

        # Load the .npz file
        datum = np.load(datum_file)
        image = datum["image"]
        flow = datum["flow"]

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            flow = self.target_transform(flow)

        # Get the caption
        caption = self.caption_data[cur_name]
        caption_emb = self.caption_embeddings[cur_name].reshape(1, -1)

        return {"image": image, "flow": flow, "caption": caption, "caption_emb": caption_emb}


class RandomPatchMasking:
    def __init__(self, mask_percentage: float = 0.4, patch_size: int = 8):
        """
        Args:
            k (float): Percentage of patches to mask (0 to 1).
        """
        self.k = mask_percentage
        self.p = patch_size

    def __call__(self, img):
        """
        Apply random patch masking to an image.

        Args:
            img (PIL Image or Tensor): Image of size (C, H, W) or (H, W).

        Returns:
            Transformed image with random patches masked.
        """
        if isinstance(img, torch.Tensor):
            pass
            # img = img.clone()  # Avoid modifying original tensor
        else:
            img = transforms.ToTensor()(img)

        _, H, _ = img.shape  # (C, 128, 128)
        patch_size = self.p
        grid_size = H // patch_size
        num_patches = grid_size**2
        num_masked = int(self.k * num_patches)  # k% of patches

        # Generate random indices for patches to mask
        indices = random.sample(range(num_patches), num_masked)

        # Mask selected patches
        for idx in indices:
            row = (idx // grid_size) * patch_size
            col = (idx % grid_size) * patch_size
            img[:, row : row + patch_size, col : col + patch_size] = 0  # Mask with zero

        return img


class RandomCropResize:
    def __init__(self, crop_ratio, resize_size):
        """
        crop_ratio: Tuple (H, W) of crop dimensions
        resize_size: Tuple (H, W) of resized dimensions
        """
        self.crop_ratio = crop_ratio
        self.resize_size = resize_size

    def __call__(self, inputs):
        """
        inputs: Tuple (image, flow, ...) with image (C, H, W) and flow (2, H, W)

        Returns cropped and resized image, flow
        """
        image = inputs[0]
        orig_w, orig_h = image.shape[1:]
        crop_h, crop_w = [int(self.crop_ratio * x) for x in (orig_h, orig_w)]

        # Ensure crop fits inside original image
        if orig_h < crop_h or orig_w < crop_w:
            raise ValueError("Crop size should be smaller than original image size.")

        # Random crop coordinates
        x1 = random.randint(0, orig_w - crop_w)
        y1 = random.randint(0, orig_h - crop_h)

        transformed_inputs = []
        # Crop-resize image & flow tensors.
        # NOTE: For resizing flow field (bilinear interpolation), flow in normalized, no scaling required.
        for tensor in inputs:
            modified_tensor = tensor[:, y1 : y1 + crop_h, x1 : x1 + crop_w]
            modified_tensor = transforms.functional.resize(modified_tensor, self.resize_size)
            transformed_inputs.append(modified_tensor)

        return tuple(transformed_inputs)


def get_joint_transforms(
    image_size=(128, 128), add_color_jitter=False, mask_args=None, crop_ratio=None, mask_crop_ratio=None
):
    """
    Returns a single transform that takes (image, flow, more_flow) and applies the same
    augmentations while ensuring consistency.
    """
    # Convert numpy to torch tensor if necessary.
    tensor_transform = transforms.Lambda(lambda x: tuple([torch.from_numpy(x_i) for x_i in x]))

    # Crop Resize Transform.
    crop_resize_transform = transforms.Lambda(
        lambda x: (RandomCropResize(crop_ratio=crop_ratio, resize_size=image_size)(x))
    )

    # Spatial transformation (resize applied to both image and flow).
    resize_transform = transforms.Lambda(lambda x: tuple([transforms.Resize(image_size)(x_i) for x_i in x]))

    # Color jitter (applied only to the image).
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    color_transform = transforms.Lambda(lambda x: (color_jitter(x[0]), *x[1:]))  # Only modifies image

    # Apply masking if specified.
    mask_transform = transforms.Lambda(lambda x: (RandomPatchMasking(**mask_args)(x[0]), *x[1:]))

    # Crop of masking transform.
    def crop_or_mask(input_sequence):
        pick = random.random() < mask_crop_ratio
        if pick:
            input_sequence = resize_transform(input_sequence)
            return mask_transform(input_sequence)
        else:
            return crop_resize_transform(input_sequence)

    crop_or_mask_transform = transforms.Lambda(crop_or_mask)

    # Compose all transforms into a pipeline.
    train_transform_list = [
        tensor_transform,
    ]
    if add_color_jitter:
        train_transform_list.append(color_transform)
    if mask_crop_ratio is not None:
        train_transform_list.append(crop_or_mask_transform)
    else:
        if crop_ratio is not None:
            train_transform_list.append(crop_resize_transform)
        else:
            train_transform_list.append(resize_transform)
        if mask_args is not None:
            train_transform_list.append(mask_transform)

    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose([tensor_transform, resize_transform])

    return train_transform, val_transform


def get_transforms(image_size=(128, 128), add_color_jitter=False, mask_args=None):
    image_transform_list = [
        transforms.Lambda(lambda x: torch.from_numpy(x)),
        transforms.Resize(image_size),
    ]
    # Val Image Transform (no augs).
    val_image_transform = transforms.Compose(image_transform_list)
    if add_color_jitter:
        image_transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        )

    if mask_args is not None:
        image_transform_list.append(RandomPatchMasking(**mask_args))

    # Final Image (input) Transform.
    image_transform = transforms.Compose(image_transform_list)

    # Final Flow (target) Transform.
    flow_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.from_numpy(x)),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    )

    return image_transform, flow_transform, val_image_transform


# Example usage
if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_ABC_D/robot_training"
    CAPTION_FILE = os.path.join(DATA_ROOT, "captions.json")
    ACTION_FILE = os.path.join(DATA_ROOT, "relative_actions.json")

    dataset = RobotTrainingDataset(DATA_ROOT, CAPTION_FILE, ACTION_FILE, include_captions=True, include_actions=True)

    for i in range(len(dataset)):
        sample = dataset[i]
        print(
            f"Sample {i}: Image shape: {sample['image'].shape}, "
            f"Flow shape: {sample['flow'].shape}, "
            f"Caption: {sample['caption']}, "
            f"Caption Emb: {sample['caption_emb'].shape}, "
            f"Relative Action: {sample['relative_action'].shape}, "
        )
        if i == 2:  # Display first 3 samples
            break
