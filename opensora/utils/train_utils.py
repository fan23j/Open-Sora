import math
import random
import torch
import wandb

from typing import List, Tuple, Optional
from logging import Logger
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from nba.src.entities.clip_annotations import ClipAnnotationWrapper
from opensora.utils.config_entity import TrainingConfig
from opensora.utils.model_helpers import calculate_weight_norm, push_to_device
from opensora.models.stdit.stdit2 import STDiT2


def log_progress(
    logger: Logger,
    sample_annotation_wrapper: ClipAnnotationWrapper,
    coordinator: DistCoordinator,
    global_step: int,
    cfg: TrainingConfig,
    writer: SummaryWriter,
    loss: torch.Tensor,
    running_loss: float,
    log_step: int,
    model: STDiT2,
    iteration_times: List[float],
    optimizer: HybridAdam,
    epoch: int,
    acc_step: int,
) -> Tuple[Optional[float], Optional[int]]:
    """
    Log the current state of the training run to W&B.
    """
    
    avg_loss = running_loss / log_step
    running_loss = 0
    log_step = 0
    writer.add_scalar("loss", loss.item(), global_step)
    weight_norm = calculate_weight_norm(model)
    
    # if this not the main process or not at log interval, return
    if not coordinator.is_master() or not  global_step % cfg.log_every == 0:
        return avg_loss, log_step
    
    if not cfg.wandb:
        return avg_loss, log_step

    wandb.log(
        {
            "avg_iteration_time": sum(iteration_times) / len(iteration_times),
            "iter": global_step,
            "epoch": epoch,
            "loss": loss.item(),
            "avg_loss": avg_loss,
            "acc_step": acc_step,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_norm": weight_norm,
        },
        step=global_step,
    )
    iteration_times.clear()
    return running_loss, log_step


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    decay: float = 0.9999,
    sharded: bool = True,
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
            # HACK
            continue
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.working_to_master_param[param_id]
                # master_param = optimizer._param_group.working_to_master_param[param_id]
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
        # print(f"mask ratios: {mask_ratios}")
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
