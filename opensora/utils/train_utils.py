import math
import random
from collections import OrderedDict

import torch
import cv2
import numpy as np


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
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
                # use either depending on the collosalai version
                master_param = optimizer.working_to_master_param[param_id]
                #master_param = optimizer._param_store.working_to_master_param[param_id]
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


def create_video_comparison_callback(save_dir, global_step):
    def save_comparison_video(x_start, x_t, t, save_dir=save_dir, global_step=global_step):
        # Denormalize the tensors
        x_start = denormalize(x_start)
        x_t = denormalize(x_t)

        # Convert to float32 if necessary and move to CPU
        x_start = x_start.float().cpu()
        x_t = x_t.float().cpu()

        # Convert to NumPy arrays
        x_start_np = x_start.detach().numpy()
        x_t_np = x_t.detach().numpy()

        # Assuming x_start and x_t are in the format [batch, channels, frames, height, width]
        batch_size, channels, num_frames, height, width = x_start_np.shape

        for i in range(batch_size):
            # Create a side-by-side comparison video
            comparison = np.zeros((height, width * 2, 3, num_frames))
            
            # Convert videos to RGB format
            x_start_rgb = convert_to_rgb(x_start_np[i])
            x_t_rgb = convert_to_rgb(x_t_np[i])
            
            comparison[:, :width, :, :] = x_start_rgb
            comparison[:, width:, :, :] = x_t_rgb

            # Clip values to [0, 1] range
            comparison = np.clip(comparison, 0, 1)

            # Convert to uint8
            comparison = (comparison * 255).astype(np.uint8)

            # Save the comparison video
            filename = f"{save_dir}/comparison_video_batch{i}_t{t[i].item()}_{global_step}.mp4"
            save_video(comparison, filename)
            break

    return save_comparison_video

def denormalize(tensor):
    """
    Denormalize the tensor that was normalized with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
    """
    return tensor * 0.5 + 0.5

def convert_to_rgb(video):
    # Assuming video shape is (channels, frames, height, width)
    if video.shape[0] == 4:
        # If 4 channels, assume RGBA and convert to RGB
        rgb = video[:3, ...]
    elif video.shape[0] == 1:
        # If 1 channel, assume grayscale and repeat to create RGB
        rgb = np.repeat(video, 3, axis=0)
    else:
        # If already 3 channels, assume it's already RGB
        rgb = video
    
    # Ensure the shape is (height, width, 3, frames)
    return np.transpose(rgb, (2, 3, 0, 1))

def save_video(video_frames, filename):
    # Assuming video_frames is in the format [height, width, channels, frames]
    height, width, channels, num_frames = video_frames.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = video_frames[:, :, :, i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()