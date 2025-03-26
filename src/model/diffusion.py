import diffusers
import numpy as np
import torch


def get_conditional_unet(image_size=128, pretrained=None, in_channels=5, out_channels=2, condition_dim=768, size="B"):
    if pretrained is not None:
        model = diffusers.UNet2DConditionModel.from_pretrained(pretrained, use_safetensors=True)
        return model

    if size == "B":
        model = diffusers.UNet2DConditionModel(
            sample_size=image_size,  # Target image resolution
            in_channels=in_channels,  # Number of input channels, e.g., RGB image + flow
            out_channels=out_channels,  # Number of output channels, e.g., flow image
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
            cross_attention_dim=condition_dim,  # Dimension of the conditional embedding (e.g., text embeddings)
        )
    elif size == "S":
        model = diffusers.UNet2DConditionModel(
            sample_size=image_size,  # Target image resolution
            in_channels=in_channels,  # Number of input channels, e.g., RGB image + flow
            out_channels=out_channels,  # Number of output channels, e.g., flow image
            layers_per_block=2,  # Number of ResNet layers per UNet block
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D"),
            cross_attention_dim=condition_dim,  # Dimension of the conditional embedding (e.g., text embeddings)
        )
    return model


def get_conditional_unet_3d(image_size=128, pretrained=None, in_channels=5, condition_dim=768, use_flash_attn=True):
    """Working setup for 3D UNet"""
    if pretrained is not None:
        model = diffusers.UNet3DConditionModel.from_pretrained(pretrained, use_safetensors=True)
        return model

    cross_attn_down = "CrossAttnDownBlock3D"  # CrossAttnDownBlockSpatioTemporal, CrossAttnDownBlock3D
    cross_attn_up = "CrossAttnUpBlock3D"  # CrossAttnUpBlockSpatioTemporal, CrossAttnUpBlock3D
    model = diffusers.UNet3DConditionModel(
        sample_size=image_size,
        in_channels=in_channels,  # RGB + OF = 5
        out_channels=2,  # Predict optical flow
        down_block_types=("DownBlock3D", "DownBlock3D", cross_attn_down, "DownBlock3D"),
        up_block_types=("UpBlock3D", cross_attn_up, "UpBlock3D", "UpBlock3D"),
        block_out_channels=(64, 128, 256, 256),
        layers_per_block=2,
        cross_attention_dim=condition_dim,
        attention_head_dim=4,
    )
    if not use_flash_attn:
        model.set_attn_processor(diffusers.models.attention_processor.AttnProcessor())

    return model


if __name__ == "__main__":
    DATA_ROOT = "/home/kanchana/data/calvin/task_D_D/robot_training"
    IMAGE_SIZE = 128

    datum_file = f"{DATA_ROOT}/eps_00000.npz"
    datum = np.load(datum_file)

    use_3D = True
    WITH_GRAD = True

    if not use_3D:
        unet_model = get_conditional_unet(IMAGE_SIZE, size="S").cuda()

        image = torch.ones((8, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
        flow = torch.ones((8, 2, IMAGE_SIZE, IMAGE_SIZE)).cuda()
        text_emb = torch.ones((8, 1, 768)).cuda()
        time_step = 0

        model_input = torch.concat([image, flow], dim=1)

        with torch.no_grad():
            pred = unet_model(model_input, time_step, text_emb).sample

        print(f"pred shape: {pred.shape}")

    else:
        IMAGE_SIZE = 64
        FRAMES = 8
        TEXT_DIM = 512
        BATCH_SIZE = 8
        unet_model = get_conditional_unet_3d(IMAGE_SIZE, condition_dim=TEXT_DIM, use_flash_attn=True).cuda()

        image = torch.ones((BATCH_SIZE, 3, FRAMES, IMAGE_SIZE, IMAGE_SIZE)).cuda()
        flow = torch.ones((BATCH_SIZE, 2, FRAMES, IMAGE_SIZE, IMAGE_SIZE)).cuda()

        text_emb = torch.ones((BATCH_SIZE, 1, TEXT_DIM)).cuda()
        time_step = 0  # torch.tensor([500], dtype=torch.long).cuda()
        model_input = torch.concat([image, flow], dim=1)
        if WITH_GRAD:
            pred = unet_model(model_input, time_step, encoder_hidden_states=text_emb).sample
        else:
            with torch.no_grad():
                pred = unet_model(model_input, time_step, encoder_hidden_states=text_emb).sample
        print(f"pred shape: {pred.shape}")

    breakpoint()
