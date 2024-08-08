import math
import random
from collections import OrderedDict
from typing import List, Dict
import torch
from colossalai.utils import get_current_device

action_classes = [
    "a basketball player missing a three-point shot",
    "a basketball player assisting on a play",
    "a basketball player setting a screen",
    "a basketball player grabbing a rebound",
    "a basketball player committing a turnover",
    "a basketball player making a free throw",
    "a basketball player missing a free throw",
    "a basketball player scoring and being fouled",
    "a basketball player missing a two-point shot",
    "a basketball player making a two-point shot",
    "a basketball player committing a foul",
    "a basketball player executing a pick and roll",
    "a basketball player posting up",
    "a basketball player stealing the ball",
    "a basketball player receiving a technical foul",
    "a basketball player making a three-point shot",
    "a basketball player committing their second foul",
    "a basketball player committing their third foul",
    "a basketball player committing an unsportsmanlike foul",
    "a basketball player making a three-pointer and being fouled",
    "a basketball player getting a second chance opportunity",
    "a basketball player making two free throws",
    "a basketball player missing two free throws",
    "a basketball player making three free throws",
    "a basketball player missing three free throws",
    "a basketball player committing a disqualifying foul"
]

description_to_index = {desc: idx for idx, desc in enumerate(action_classes)}

def get_text_encodings(
    prompts: List[str],
    model_args: Dict[str, torch.Tensor],
    is_train: bool = True
):
    device = get_current_device()
    indices = [description_to_index[desc] for desc in prompts if desc in description_to_index]
    y_filtered = model_args["y"][indices]
    mask_filtered = model_args["mask"][indices]
    
    if not is_train:
        null_encodings = model_args["y"][26:26 + len(prompts)]
        y_filtered = torch.cat([y_filtered, null_encodings], dim=0)

    filtered_model_args = {
        "y": y_filtered.to(device),
        "mask": mask_filtered.to(device)
    }
    
    return filtered_model_args


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMa model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.working_to_master_param[param_id]
                #master_param = optimizer._param_group.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "mask_no",
            "mask_quarter_random",
            "mask_quarter_head",
            "mask_quarter_tail",
            "mask_quarter_head_tail",
            "mask_image_random",
            "mask_image_head",
            "mask_image_tail",
            "mask_image_head_tail",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        print(f"mask ratios: {mask_ratios}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "mask_quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "mask_image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "mask_quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "mask_image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "mask_quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "mask_image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks
