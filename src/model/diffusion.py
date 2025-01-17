import diffusers
import numpy as np
import torch


def get_conditional_unet(image_size=128):
    model = diffusers.UNet2DConditionModel(
        sample_size=image_size,  # Target image resolution
        in_channels=5,  # Number of input channels, e.g., RGB image + flow
        out_channels=2,  # Number of output channels, e.g., flow image
        layers_per_block=2,  # Number of ResNet layers per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # Channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # Regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # Downsampling block with attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # Regular ResNet upsampling block
            "AttnUpBlock2D",  # Upsampling block with attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=768,  # Dimension of the conditional embedding (e.g., text embeddings)
    )

    return model


if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    IMAGE_SIZE = 128

    datum_file = f"{DATA_ROOT}/eps_00000.npz"
    datum = np.load(datum_file)

    unet_model = get_conditional_unet(IMAGE_SIZE).cuda()

    # image = torch.Tensor(datum["image"]).cuda()
    image = torch.ones((8, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    flow = torch.ones((8, 2, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    text_emb = torch.ones((8, 1, 768)).cuda()
    time_step = 0

    model_input = torch.concat([image, flow], dim=1)

    with torch.no_grad():
        pred = unet_model(model_input, time_step, text_emb).sample

    print(f"pred shape: {pred.shape}")
