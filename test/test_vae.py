from io import BytesIO

import numpy as np
import requests
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

_IMAGE_SIZE = (512, 512)


# Image preprocessing
def preprocess_image(img, target_size=_IMAGE_SIZE):
    transform = Compose([
        Resize(target_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension


# Load the SDXL VAE model
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

# Load and preprocess the image
img_url = "https://epipoca.com.br/wp-content/uploads/2021/03/cf281f60a52896f2914116b85c74b809.jpg"
response = requests.get(img_url)
img = Image.open(BytesIO(response.content)).resize(_IMAGE_SIZE)
img_tensor = preprocess_image(img)


# Move to device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
img_tensor = img_tensor.to(device)

# Encode the image
with torch.no_grad():
    latent = vae.encode(img_tensor).latent_dist.sample()  # Sample latent code

# Decode the latent representation back to an image
with torch.no_grad():
    recon_img = vae.decode(latent).sample

# Post-process the reconstructed image to [0, 255] range
recon_vis = ((recon_img.squeeze(0).permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
orig_vis = np.array(img)

vis_pair = Image.fromarray(np.concatenate([orig_vis,recon_vis], axis=1))
# vis_pair.show()  # Uncomment to display the image

assert torch.abs(recon_img - img_tensor).mean() < 0.05, "Reconstruction L1 loss above 0.05"