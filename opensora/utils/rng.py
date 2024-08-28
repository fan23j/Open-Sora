import torch
import random
import numpy as np


def save_rng_state():
    rng_state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    return rng_state


def load_rng_state(rng_state):
    torch.set_rng_state(rng_state["torch"])
    if rng_state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["random"])


# from mmengine.runner import set_random_seed
def set_seed_custom(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # set_random_seed(seed=seed)
