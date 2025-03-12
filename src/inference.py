import torch
from diffusers import DDIMScheduler

from src.model import diffusion


# Define the inference function
def run_inference(model, start_flow, vis_cond, text_cond, num_inference_steps=50):
    """
    Run inference using a DDIM scheduler.

    Args:
        model: The custom model to be used for inference.
        start_flow: The noisy latent flow input.
        vis_cond: Visual condition inputs.
        text_cond: Text condition input.
        num_inference_steps: Number of DDIM steps to run (default: 50).

    Returns:
        final_output: The final denoised output from the model.
    """
    # Initialize the DDIM scheduler
    scheduler = DDIMScheduler(num_train_timesteps=1000)  # You can adjust timesteps based on your model's training
    scheduler.set_timesteps(num_inference_steps)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the noisy input as the latent
    latents = start_flow.clone()

    # Iterate through DDIM timesteps
    for t in scheduler.timesteps:
        # Prepare the model inputs
        with torch.no_grad():
            # Predict the noise (epsilon) using the model
            model_input = torch.concat([latents, vis_cond], dim=1)
            time_step = torch.ones(latents.shape[0], dtype=torch.int64, device=latents.device) * t
            predicted_noise = model(model_input, time_step, text_cond, return_dict=False)[0]

        # Update the latent based on DDIM step
        latents = scheduler.step(predicted_noise, t, latents).prev_sample

    # The final latent is the denoised output
    final_output = latents
    return final_output


# Example usage
if __name__ == "__main__":
    # Dummy inputs for demonstration purposes
    image_size = (128, 128)
    noisy_flow = torch.randn(1, 2, *image_size).cuda()  # Replace with actual noisy input
    image_cond = torch.randn(1, 3, *image_size).cuda()  # Replace with actual image condition
    text_cond = torch.randn(1, 1, 768).cuda()  # Replace with actual text condition (e.g., text embeddings)

    model = diffusion.get_conditional_unet(image_size[0]).cuda()

    # Run inference
    final_output = run_inference(model, noisy_flow, image_cond, text_cond, num_inference_steps=50)

    print("Inference complete. Output shape:", final_output.shape)
