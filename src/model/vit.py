import torch
import transformers
from torchvision.models import vision_transformer


def get_vit_tiny_hf(image_size=128, patch_size=16, in_channels=5, action_space=7):
    # Model Settings.
    hidden_dim = 192
    num_layers = 12
    num_heads = 3
    mlp_dim = 4 * hidden_dim

    model_config = transformers.ViTConfig(
        hidden_size=hidden_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=mlp_dim,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=in_channels,
        num_labels=action_space,
    )
    model = transformers.ViTForImageClassification(model_config)

    return model


def get_vit_base_hf(image_size=128, patch_size=16, in_channels=5):
    # Model Settings for ViT-Base.
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    mlp_dim = 4 * hidden_dim
    action_space = 7

    model_config = transformers.ViTConfig(
        hidden_size=hidden_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=mlp_dim,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=in_channels,
        num_labels=action_space,
    )
    model = transformers.ViTForImageClassification(model_config)

    return model


def get_vit_tiny(image_size=128, patch_size=16, in_channels=5):
    # Model Settings.
    hidden_dim = 192
    num_layers = 12
    num_heads = 3
    mlp_dim = 4 * hidden_dim

    # Load a Vision Transformer Base model (to modify into Tiny)
    model = vision_transformer.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=7,
    )
    # Update model input channels.
    model.conv_proj = torch.nn.Conv2d(
        in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
    )

    return model


if __name__ == "__main__":
    vit_tiny = get_vit_tiny()
    vit_tiny.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 128
    input_channels = 5
    dummy_input = torch.randn(1, 5, img_size, img_size)  # 1 image, 5 channels, 128x128
    output = vit_tiny(dummy_input)

    print(output.shape)  # Default output [1, 7] (for action vectors)

    # Huggingface model.
    vit_tiny_hf = get_vit_tiny_hf()
    vit_tiny_hf.eval()

    # Test with a variable image size (e.g., 128x128)
    img_size = 128
    input_channels = 5
    dummy_input = torch.randn(1, 5, img_size, img_size)  # 1 image, 5 channels, 128x128
    output = vit_tiny_hf(dummy_input)

    print(output.logits.shape)  # Default output [1, 7] (for action vectors)
