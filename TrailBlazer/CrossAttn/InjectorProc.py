from typing import Dict, List, TypedDict
import numpy as np
import torch
import math
import torch.nn as nn

from ..Misc import Logger as log

from ..Misc.BBox import BoundingBox

KERNEL_DIVISION = 3.
INJECTION_SCALE = 1.0

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    gaussian_map.div_(gaussian_map.max())
    return gaussian_map

class InjectorProcessor(nn.Module):
    def __init__(
        self,
        bbox_per_frame: List[BoundingBox],
        strengthen_scale: float = 0.0,
        weaken_scale: float = 1.0,
        is_spatial: bool = False,
    ):
        super().__init__()
        self.prompt = "A basketball player missing a three-point shot"
        base_prompt = self.prompt.split(";")[0]
        self.len_prompt = len(base_prompt.split(" "))
        self.prompt_len = len(self.prompt.split(" "))
        self.strengthen_scale = strengthen_scale
        self.weaken_scale = weaken_scale
        self.num_frames = len(bbox_per_frame)
        self.bbox_per_frame = bbox_per_frame
        self.use_weaken = True
        self.is_spatial = is_spatial
        
    def localized_weight_map(self, attention_probs_4d, token_inds, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_4d.size()[1])
        max_val = attention_probs_4d.max()
        weight_map = torch.zeros_like(attention_probs_4d).half()
        frame_size = attention_probs_4d.shape[0] // len(bbox_per_frame)
        for i in range(len(bbox_per_frame)):
            bbox_ratios = bbox_per_frame[i]
            bbox = BoundingBox(dim_x, dim_y, bbox_ratios)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(frame_size, 1, 1, len(token_inds))
                .to(attention_probs_4d.device)
            ).half()

            scale = attention_probs_4d.max() * INJECTION_SCALE
            noise_patch.mul_(scale)

            b_idx = frame_size * i
            e_idx = frame_size * (i + 1)
            bbox.sliced_tensor_in_bbox(weight_map)[
                b_idx:e_idx, ..., token_inds
            ] = noise_patch
        return weight_map

    def localized_temporal_weight_map(self, attention_probs_5d, bbox_per_frame, dim_x, dim_y, scale=1):
        """Using guassian 2d distribution to generate weight map and return the
        array with the same size of the attention argument.
        """
        dim = int(attention_probs_5d.size()[1])
        f = attention_probs_5d.shape[-1]
        max_val = attention_probs_5d.max()
        weight_map = torch.zeros_like(attention_probs_5d).half()

        def get_patch(bbox_at_frame, i, j, bbox_per_frame):
            bbox = BoundingBox(dim_x, dim_y, bbox_at_frame)
            # Generating the gaussian distribution map patch
            x = torch.linspace(0, bbox.height, bbox.height)
            y = torch.linspace(0, bbox.width, bbox.width)
            x, y = torch.meshgrid(x, y, indexing="ij")
            noise_patch = (
                gaussian_2d(
                    x,
                    y,
                    mx=int(bbox.height / 2),
                    my=int(bbox.width / 2),
                    sx=float(bbox.height / KERNEL_DIVISION),
                    sy=float(bbox.width / KERNEL_DIVISION),
                )
                .unsqueeze(0)
                .repeat(attention_probs_5d.shape[0], 1, 1)
                .to(attention_probs_5d.device)
            ).half()
            scale = attention_probs_5d.max() * INJECTION_SCALE
            noise_patch.mul_(scale)
            inv_noise_patch = noise_patch - noise_patch.max()
            dist = (float(abs(j - i))) / len(bbox_per_frame)
            final_patch = inv_noise_patch * dist + noise_patch * (1. - dist)
            #final_patch = noise_patch * (1. - dist)
            #final_patch = inv_noise_patch * dist
            return final_patch, bbox


        for j in range(len(bbox_per_frame)):
            for i in range(len(bbox_per_frame)):
                patch_i, bbox_i = get_patch(bbox_per_frame[i], i, j, bbox_per_frame)
                patch_j, bbox_j = get_patch(bbox_per_frame[j], i, j, bbox_per_frame)
                bbox_i.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_i
                bbox_j.sliced_tensor_in_bbox(weight_map)[..., i, j] = patch_j

        return weight_map

    def forward(self, attention_probs: torch.Tensor, dim_x, dim_y):
        """ """
        frame_size = attention_probs.shape[0] // self.num_frames
        num_affected_frames = self.num_frames
        attention_probs_copied = attention_probs.detach().clone()

        trailing_length = 100
        trailing_inds = list(
            range(self.len_prompt + 1, self.len_prompt + trailing_length + 1)
        )
        # NOTE: Spatial cross attention editing
        if self.is_spatial:
            token_inds = [2,3]
            all_tokens_inds = list(set(token_inds).union(set(trailing_inds)))
            strengthen_map = self.localized_weight_map(
                attention_probs_copied,
                token_inds=all_tokens_inds,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y
            )

            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale

            # weakening
            attention_probs_copied[..., all_tokens_inds] *= weaken_map[
                ..., all_tokens_inds
            ]
            # strengthen
            attention_probs_copied[..., all_tokens_inds] += (
                self.strengthen_scale * strengthen_map[..., all_tokens_inds]
            )
        # NOTE: Temporal cross attention editing
        elif not self.is_spatial:
            strengthen_map = self.localized_temporal_weight_map(
                attention_probs_copied,
                bbox_per_frame=self.bbox_per_frame,
                dim_x = dim_x,
                dim_y = dim_y
            )
            weaken_map = torch.ones_like(strengthen_map)
            zero_indices = torch.where(strengthen_map == 0)
            weaken_map[zero_indices] = self.weaken_scale
            # weakening
            attention_probs_copied *= weaken_map
            # strengthen
            attention_probs_copied += self.strengthen_scale * strengthen_map

        return attention_probs_copied
