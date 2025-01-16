import numpy as np
from matplotlib import pyplot as plt


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
