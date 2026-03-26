import torch
from functools import lru_cache
from moge.model.v2 import MoGeModel


@lru_cache(maxsize=1)
def get_model(device):
    """Return the same MoGe model instance every time."""
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()                    # helpful if you only infer
    return model


def get_depth_map(img, model, f=None):
    """
    Extract the depth map from the model output.
    `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
    The maps are in the same size as the input image. 
    {
        "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
        "depth": (H, W),        # depth map
        "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
        "mask": (H, W),         # a binary mask for valid pixels. 
        "intrinsics": (3, 3),   # normalized camera intrinsics
    }

    Args:
        img: Input image as a torch tensor of shape (3, H, W) with RGB values normalized to [0, 1].
    Returns:
        depth_map: Depth map of shape (H, W).
    """
    if f is not None:
        W = img.shape[2]
        fov_x = torch.rad2deg(2 * torch.atan(W / (2 * f)))
    else:
        fov_x = None
    output = model.infer(img, fov_x=fov_x)
    return output['depth']


def depth_bilinear(depth, u, v):
    H, W = depth.shape
    uu = 2.0 * (u / (W - 1)) - 1.0  # normalise to [‑1,1]
    vv = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack((uu, vv), -1).view(1, -1, 1, 2)  # [1, N, 1, 2]
    return torch.nn.functional.grid_sample(depth.view(1, 1, H, W), grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=True).view(-1)


def sample_depth_map(depth_map, indices):
    """
    Sample the depth map at given indices.
    
    Args:
        depth_map: Depth map of shape (H, W) (torch tensor).
        indices: Torch tensor of shape (N, 2) where the first dimension is the column
                 index and the second dimension is the row index.

    Returns:
        sampled_depths: Torch tensor of shape (N,) containing the sampled depth values.
    """
    x = indices[:, 0]  # Column indices
    y = indices[:, 1]  # Row indices
    sampled_depths = depth_bilinear(depth_map, x, y)
    return sampled_depths
