import torch
from torchvision.models import vision_transformer


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
