import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image


class FlowNormalizer:
    def __init__(self, height, width):
        """
        Initialize the FlowNormalizer with the dimensions of the flow.

        Args:
            height (int): The height (H) of the flow.
            width (int): The width (W) of the flow.
        """
        self.height = height
        self.width = width

    def normalize(self, flow):
        """
        Normalize the flow tensor to the range [0, 1].

        Args:
            flow (numpy.ndarray): The flow tensor of shape (B, H, W, 2).

        Returns:
            numpy.ndarray: The normalized flow tensor of the same shape.
        """
        normalized_flow = np.zeros_like(flow, dtype=np.float32)
        # Normalize channel 0 (height)
        normalized_flow[..., 0] = (flow[..., 0] + self.height) / (2 * self.height)
        # Normalize channel 1 (width)
        normalized_flow[..., 1] = (flow[..., 1] + self.width) / (2 * self.width)
        return normalized_flow

    def unnormalize(self, normalized_flow):
        """
        Unnormalize the flow tensor from the range [0, 1] back to the original range.

        Args:
            normalized_flow (numpy.ndarray): The normalized flow tensor of shape (B, H, W, 2).

        Returns:
            numpy.ndarray: The unnormalized flow tensor of the same shape.
        """
        flow = np.zeros_like(normalized_flow, dtype=np.float32)
        # Unnormalize channel 0 (height)
        flow[..., 0] = (normalized_flow[..., 0] * 2 * self.height) - self.height
        # Unnormalize channel 1 (width)
        flow[..., 1] = (normalized_flow[..., 1] * 2 * self.width) - self.width
        return flow


def visualize_flow_vectors_as_PIL(image, flow, step=16, title="Optical Flow Vectors"):
    """
    Overlay optical flow vectors on an image.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3).
        flow (numpy.ndarray): Optical flow array (H, W, 2).
        step (int): Sampling step for displaying flow vectors.

    Returns:
        PIL.Image.Image: Visualization as a PIL image.
    """

    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(np.int32)
    fx, fy = flow[x, y].T

    # Create a matplotlib figure
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Overlay flow vectors
    ax.imshow(image)
    ax.quiver(x, y, fx, fy, color="red", angles="xy", scale_units="xy", scale=1, width=0.002)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    # Render the figure to a PIL image
    canvas.draw()
    buf = canvas.buffer_rgba()
    pil_image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)

    return pil_image


def visualize_flow_vectors(image, flow, step=16, save_path=None, title="Optical Flow Vectors"):
    """
    Overlay optical flow vectors on an image.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3).
        flow (numpy.ndarray): Optical flow array (H, W, 2).
        step (int): Sampling step for displaying flow vectors.

    Returns:
        None (displays the visualization).
    """
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step].astype(np.int32)
    fx, fy = flow[x, y].T

    # Overlay flow vectors
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.quiver(x, y, fx, fy, color="red", angles="xy", scale_units="xy", scale=1, width=0.002)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_multiple_flow_vectors_as_PIL(image, flows, step=16, title="Optical Flow Vectors (Future Steps)"):
    """
    Overlay multiple optical flow vectors on an image for several future steps.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3).
        flows (numpy.ndarray): Optical flow array (T, H, W, 2), where T is the number of steps.
        step (int): Sampling step for displaying flow vectors.

    Returns:
        PIL.Image.Image: Visualization as a PIL image.
    """
    T, H, W, _ = flows.shape
    y, x = np.mgrid[step // 2 : H : step, step // 2 : W : step].astype(np.int32)

    # Create matplotlib figure
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.imshow(image)
    colors = plt.cm.viridis(np.linspace(0, 1, T))  # Unique color for each step

    for t in range(T):
        fx, fy = flows[t, x, y].T
        ax.quiver(
            x,
            y,
            fx,
            fy,
            color=colors[t],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            alpha=0.8,
            label=f"Step {t + 1}",
        )

    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize="small", frameon=True)
    fig.tight_layout()

    # Render to PIL image
    canvas.draw()
    buf = canvas.buffer_rgba()
    pil_image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)

    return pil_image


def flow_to_pil_hsv(flow_tensor, saturation=255, gamma=2.0):
    """
    Convert a flow field (C=2, H, W) torch.Tensor to a color image using PIL.
    Returns a PIL RGB Image.
    """
    if flow_tensor.shape[0] != 2:
        raise ValueError("Expected flow tensor shape (2, H, W)")

    if isinstance(flow_tensor, np.ndarray):
        flow = flow_tensor
    else:
        flow = flow_tensor.detach().cpu().numpy()
    fx, fy = flow[0], flow[1]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)  # range: [-pi, pi]

    # Normalize angle to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)
    # Normalize magnitude to [0, 1] for value
    max_magnitude = max(np.percentile(magnitude, 99), 5)
    mag_norm = np.clip(magnitude / (max_magnitude + 1e-5), 0, 1)
    mag_norm = np.power(mag_norm, gamma)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = (hue * 255).astype(np.uint8)  # Hue

    hsv[..., 1] = (saturation * mag_norm).astype(np.uint8)
    hsv[..., 2] = (255 * mag_norm + (1 - mag_norm) * 255).astype(np.uint8)

    hsv_image = Image.fromarray(hsv, mode="HSV")
    rgb_image = hsv_image.convert("RGB")
    return rgb_image


# Example usage
if __name__ == "__main__":
    B, H, W = 2, 10, 15  # Example batch size, height, and width
    flow = np.random.uniform(-H, H, size=(B, H, W, 2))
    flow[..., 1] = np.random.uniform(-W, W, size=(B, H, W))  # Ensure channel 1 range

    normalizer = FlowNormalizer(H, W)
    normalized_flow = normalizer.normalize(flow)
    unnormalized_flow = normalizer.unnormalize(normalized_flow)

    print("Original Flow:", flow)
    print("Normalized Flow:", normalized_flow)
    print("Unnormalized Flow:", unnormalized_flow)

    # Check if unnormalization matches the original flow
    assert np.allclose(flow, unnormalized_flow, atol=1e-4), "Unnormalized flow does not match the original!"
