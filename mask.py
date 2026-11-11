import torch
from Mask_generator import gaussian
import numpy as np


class mask_generator:
    def __init__(self, shape, seed, accelerate, acs_lines, type="gaussian"):
        self.shape = shape
        self.seed = seed
        self.accelerate = accelerate
        self.acs_lines = acs_lines
        self.type = type

    def __call__(self, random_state=None):
        """
        random_state: control the random state for mask generation
        """
        if self.type == "gaussian":
            mask = gaussion_1d(self.shape, self.seed, self.accelerate, self.acs_lines, random_state)
        else:
            mask = equal_1d(self.shape, self.seed, self.accelerate, self.acs_lines, random_state)
        return mask


def equal_1d(shape: tuple, seed: int, accelerate :int, acs_lines: int, random_state:int|None =None):
    """
    return numpy(1, 1, 1, h)
    """
    h = shape[1]
    h_start = (h - acs_lines) // 2
    mask = np.zeros(h, dtype=np.float64)      
    mask[h_start: h_start + acs_lines] = 1
    if random_state is not None:
        mask[random_state % (accelerate-1) + 1 :: accelerate] = 1 
        # mask[::accelerate] = 1 
    else:
        mask[::accelerate] = 1 
    mask = np.expand_dims(mask, axis=0) * np.ones(shape)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return mask

def gaussion_1d(shape: tuple, seed: int, accelerate :int, acs_lines: int, random_state:int| None=None):
    """
    return numpy(1, 1, 1, h)
    """
    h = shape[1]
    h_start = (h - acs_lines) // 2
    mask = np.zeros(h, dtype=np.float64)       
    nonzero_count = int(np.round(h / accelerate - acs_lines)) - 1
    mask[h_start: h_start + acs_lines] = 1
      
    if random_state is not None:
        gaussian.gaussian_mask_1d(
                nonzero_count, h, h // 2,  6 * np.sqrt(h // 2), mask,  seed + random_state)
    else:
        gaussian.gaussian_mask_1d(
                nonzero_count, h, h // 2, 6 * np.sqrt(h // 2), mask,  seed) 
    mask = np.expand_dims(mask, axis=0) * np.ones(shape)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return mask
