import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass

from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet3DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
try:
    from diffusers.pipeline_utils import DiffusionPipeline
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..CrossAttn.InjectorProc import InjectorProcessor
from ..Misc import Logger as log
from ..Misc import Const


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def use_dd_temporal(model, use=True):
    """ To determine using the temporal attention editing at a step
    """
    for name, module in model.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn" in name:
            module.processor.use_dd_temporal = use


def use_dd(model, use=True):
    """ To determine using the spatial attention editing at a step
    """
    for name, module in model.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and "attn" in name:
            module.processor.use_dd = use


def initialization(model, bbox_per_frame):
    spatial_strengthen_scale = 0.125
    spatial_weaken_scale = 0.001
    temp_strengthen_scale = 0.125
    temp_weaken_scale = 0.001

    for name, module in model.named_modules():
        module_name = type(module).__name__

        if module_name == "Attention" and "attn" in name:
            if "attn_temp" in name:
                processor = InjectorProcessor(
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=temp_strengthen_scale,
                    weaken_scale=temp_weaken_scale,
                    is_text2vidzero=False,
                    name=name,
                )
            else:
                processor = InjectorProcessor(
                    bbox_per_frame=bbox_per_frame,
                    strengthen_scale=spatial_strengthen_scale,
                    weaken_scale=spatial_weaken_scale,
                    is_text2vidzero=False,
                    name=name,
                )
            module.processor = processor
            # print(name)
        
    use_dd(model, use=True)
    use_dd_temporal(model, use=True)
    log.info("Initialized")


def keyframed_prompt_embeds(bundle, encode_prompt_func, device):
    num_frames = bundle["keyframe"][-1]["frame"] + 1
    keyframe = bundle["keyframe"]
    f = lambda start, end, index: (1 - index) * start + index * end
    n = len(keyframe)
    keyed_prompt_embeds = []
    for i in range(n - 1):
        if i == 0:
            start_fr = keyframe[i]["frame"]
        else:
            start_fr = keyframe[i]["frame"] + 1
        end_fr = keyframe[i + 1]["frame"]

        start_prompt = keyframe[i]["prompt"] + Const.POSITIVE_PROMPT
        end_prompt = keyframe[i + 1]["prompt"] + Const.POSITIVE_PROMPT
        clip_length = end_fr - start_fr + 1

        start_prompt_embeds, _ = encode_prompt_func(
            start_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=Const.NEGATIVE_PROMPT,
        )

        end_prompt_embeds, negative_prompt_embeds = encode_prompt_func(
            end_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=Const.NEGATIVE_PROMPT,
        )

        for fr in range(clip_length):
            index = float(fr) / (clip_length - 1)
            keyed_prompt_embeds.append(f(start_prompt_embeds, end_prompt_embeds, index))
    assert len(keyed_prompt_embeds) == num_frames

    return torch.cat(keyed_prompt_embeds), negative_prompt_embeds.repeat_interleave(
        num_frames, dim=0
    )


def keyframed_bbox(bundle):

    keyframe = bundle["keyframe"]
    bbox_per_frame = []
    f = lambda start, end, index: (1 - index) * start + index * end
    n = len(keyframe)
    for i in range(n - 1):
        if i == 0:
            start_fr = keyframe[i]["frame"]
        else:
            start_fr = keyframe[i]["frame"] + 1
        end_fr = keyframe[i + 1]["frame"]
        start_bbox = keyframe[i]["bbox_ratios"]
        end_bbox = keyframe[i + 1]["bbox_ratios"]
        clip_length = end_fr - start_fr + 1
        for fr in range(clip_length):
            index = float(fr) / (clip_length - 1)
            bbox = []
            for j in range(4):
                bbox.append(f(start_bbox[j], end_bbox[j], index))
            bbox_per_frame.append(bbox)

    return bbox_per_frame
