#!/usr/bin/env python
"""
This script runs a Gradio App for the Open-Sora model.

Usage:
    python demo.py <config-path>
"""

import argparse
import importlib
import os
import subprocess
import sys
import re
import json
import math

import spaces
import torch

import gradio as gr
from tempfile import NamedTemporaryFile
import datetime


MODEL_TYPES = ["v1.1-stage2", "v1.1-stage3"]
CONFIG_MAP = {
    "v1.1-stage2": "configs/opensora-v1-1/inference/sample-ref.py",
    "v1.1-stage3": "configs/opensora-v1-1/inference/sample-ref.py",
}
HF_STDIT_MAP = {
    "v1.1-stage2": "hpcai-tech/OpenSora-STDiT-v2-stage2",
    "v1.1-stage3": "hpcai-tech/OpenSora-STDiT-v2-stage3",
}
RESOLUTION_MAP = {
    "144p": {
        "16:9": (256, 144),
        "9:16": (144, 256),
        "4:3": (221, 165),
        "3:4": (165, 221),
        "1:1": (192, 192),
    },
    "240p": {
        "16:9": (426, 240),
        "9:16": (240, 426),
        "4:3": (370, 278),
        "3:4": (278, 370),
        "1:1": (320, 320),
    },
    "360p": {
        "16:9": (640, 360),
        "9:16": (360, 640),
        "4:3": (554, 416),
        "3:4": (416, 554),
        "1:1": (480, 480),
    },
    "480p": {
        "16:9": (854, 480),
        "9:16": (480, 854),
        "4:3": (740, 555),
        "3:4": (555, 740),
        "1:1": (640, 640),
    },
    "720p": {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1108, 832),
        "3:4": (832, 1110),
        "1:1": (960, 960),
    },
}


# ============================
# Utils
# ============================
def collect_references_batch(reference_paths, vae, image_size):
    from opensora.datasets.utils import read_from_path

    refs_x = []
    for reference_path in reference_paths:
        if reference_path is None:
            refs_x.append([])
            continue
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size, transform_name="resize_crop")
            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    # refs_x: [batch, ref_num, C, T, H, W]
    return refs_x


def process_mask_strategy(mask_strategy):
    mask_batch = []
    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        assert (
            len(mask_group) >= 1 and len(mask_group) <= 6
        ), f"Invalid mask strategy: {mask}"
        if len(mask_group) == 1:
            mask_group.extend(["0", "0", "0", "1", "0"])
        elif len(mask_group) == 2:
            mask_group.extend(["0", "0", "1", "0"])
        elif len(mask_group) == 3:
            mask_group.extend(["0", "1", "0"])
        elif len(mask_group) == 4:
            mask_group.extend(["1", "0"])
        elif len(mask_group) == 5:
            mask_group.append("0")
        mask_batch.append(mask_group)
    return mask_batch


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i):
    masks = []
    for i, mask_strategy in enumerate(mask_strategys):
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        if mask_strategy is None:
            masks.append(mask)
            continue
        mask_strategy = process_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            loop_id = int(loop_id)
            if loop_id != loop_i:
                continue
            m_id = int(m_id)
            m_ref_start = int(m_ref_start)
            m_length = int(m_length)
            m_target_start = int(m_target_start)
            edit_ratio = float(edit_ratio)
            ref = refs_x[i][m_id]  # [C, T, H, W]
            if m_ref_start < 0:
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            z[i, :, m_target_start : m_target_start + m_length] = ref[
                :, m_ref_start : m_ref_start + m_length
            ]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    masks = torch.stack(masks)
    return masks


def process_prompts(prompts, num_loop):
    from opensora.models.text_encoder.t5 import text_preprocessing

    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                text = text_preprocessing(text)
                end_loop = (
                    int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                )
                text_list.extend([text] * (end_loop - start_loop))
            assert (
                len(text_list) == num_loop
            ), f"Prompt loop mismatch: {len(text_list)} != {num_loop}"
            ret_prompts.append(text_list)
        else:
            prompt = text_preprocessing(prompt)
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts


def extract_json_from_prompts(prompts):
    additional_infos = []
    ret_prompts = []
    for prompt in prompts:
        parts = re.split(r"(?=[{\[])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        ret_prompts.append(parts[0])
        if len(parts) == 1:
            additional_infos.append({})
        else:
            additional_infos.append(json.loads(parts[1]))
    return ret_prompts, additional_infos


# ============================
# Runtime Environment
# ============================
def install_dependencies(enable_optimization=False):
    """
    Install the required dependencies for the demo if they are not already installed.
    """

    def _is_package_available(name) -> bool:
        try:
            importlib.import_module(name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    # flash attention is needed no matter optimization is enabled or not
    # because Hugging Face transformers detects flash_attn is a dependency in STDiT
    # thus, we need to install it no matter what
    if not _is_package_available("flash_attn"):
        subprocess.run(
            f"{sys.executable} -m pip install flash-attn --no-build-isolation",
            env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
            shell=True,
        )

    if enable_optimization:
        # install apex for fused layernorm
        if not _is_package_available("apex"):
            subprocess.run(
                f'{sys.executable} -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git',
                shell=True,
            )

        # install ninja
        if not _is_package_available("ninja"):
            subprocess.run(f"{sys.executable} -m pip install ninja", shell=True)

        # install xformers
        if not _is_package_available("xformers"):
            subprocess.run(
                f"{sys.executable} -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers",
                shell=True,
            )


# ============================
# Model-related
# ============================
def read_config(config_path):
    """
    Read the configuration file.
    """
    from mmengine.config import Config

    return Config.fromfile(config_path)


def build_models(model_type, config, enable_optimization=False, model=None):
    """
    Build the models for the given model type and configuration.
    """
    # build vae
    from opensora.registry import MODELS, build_module

    vae = build_module(config.vae, MODELS).cuda()

    # build text encoder
    text_encoder = build_module(config.text_encoder, MODELS)  # T5 must be fp32
    text_encoder.t5.model = text_encoder.t5.model.cuda()

    # build stdit
    # we load model from HuggingFace directly so that we don't need to
    # handle model download logic in HuggingFace Space
    from opensora.models.stdit.stdit2 import STDiT2

    stdit = STDiT2.from_pretrained(
        HF_STDIT_MAP[model_type] if model is None else model,
        enable_flash_attn=enable_optimization,
        trust_remote_code=True,
    ).cuda()

    # build scheduler
    from opensora.registry import SCHEDULERS

    scheduler = build_module(config.scheduler, SCHEDULERS)

    # hack for classifier-free guidance
    text_encoder.y_embedder = stdit.y_embedder

    # move modelst to device
    vae = vae.to(torch.bfloat16).eval()
    text_encoder.t5.model = text_encoder.t5.model.eval()  # t5 must be in fp32
    stdit = stdit.to(torch.bfloat16).eval()

    # clear cuda
    torch.cuda.empty_cache()
    return vae, text_encoder, stdit, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="v1.1-stage3",
        choices=MODEL_TYPES,
        help=f"The type of model to run for the Gradio App, can only be {MODEL_TYPES}",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--output", default="./outputs", type=str, help="The path to the output folder"
    )
    parser.add_argument(
        "--port", default=None, type=int, help="The port to run the Gradio App on."
    )
    parser.add_argument(
        "--host", default=None, type=str, help="The host to run the Gradio App on."
    )
    parser.add_argument(
        "--share", action="store_true", help="Whether to share this gradio demo."
    )
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Whether to enable optimization such as flash attention and fused layernorm",
    )
    return parser.parse_args()


# ============================
# Main Gradio Script
# ============================
# as `run_inference` needs to be wrapped by `spaces.GPU` and the input can only be the prompt text
# so we can't pass the models to `run_inference` as arguments.
# instead, we need to define them globally so that we can access these models inside `run_inference`

# read config
args = parse_args()
config = read_config(CONFIG_MAP[args.model_type])

# make outputs dir
os.makedirs(args.output, exist_ok=True)

# disable torch jit as it can cause failure in gradio SDK
# gradio sdk uses torch with cuda 11.3
torch.jit._state.disable()

# set up
install_dependencies(enable_optimization=args.enable_optimization)

# import after installation
from opensora.datasets import IMG_FPS, save_sample
from opensora.utils.misc import to_torch_dtype

# some global variables
dtype = to_torch_dtype(config.dtype)
device = torch.device("cuda")

# build model
vae, text_encoder, stdit, scheduler = build_models(
    args.model_type,
    config,
    enable_optimization=args.enable_optimization,
    model=args.model,
)


def run_inference(mode, prompt_text, seed, cfg_scale, length):
    resolution = "360p"
    aspect_ratio = "9:16"
    # length = "2s"
    reference_image = None
    sampling_steps = 100

    torch.manual_seed(seed)
    with torch.inference_mode():
        # ======================
        # 1. Preparation
        # ======================
        # parse the inputs
        resolution = RESOLUTION_MAP[resolution][aspect_ratio]

        # gather args from config
        num_frames = config.num_frames
        frame_interval = config.frame_interval
        fps = config.fps
        condition_frame_length = config.condition_frame_length

        # compute number of loops
        if mode == "Text2Image":
            num_frames = 1
            num_loop = 1
        else:
            num_seconds = int(length.rstrip("s"))
            if num_seconds <= 16:
                num_frames = num_seconds * fps // frame_interval
                num_loop = 1
            else:
                config.num_frames = 16
                total_number_of_frames = num_seconds * fps / frame_interval
                num_loop = math.ceil(
                    (total_number_of_frames - condition_frame_length)
                    / (num_frames - condition_frame_length)
                )

        # prepare model args
        if config.num_frames == 1:
            fps = IMG_FPS

        model_args = dict()
        height_tensor = torch.tensor([resolution[0]], device=device, dtype=dtype)
        width_tensor = torch.tensor([resolution[1]], device=device, dtype=dtype)
        num_frames_tensor = torch.tensor([num_frames], device=device, dtype=dtype)
        ar_tensor = torch.tensor(
            [resolution[0] / resolution[1]], device=device, dtype=dtype
        )
        fps_tensor = torch.tensor([fps], device=device, dtype=dtype)
        model_args["height"] = height_tensor
        model_args["width"] = width_tensor
        model_args["num_frames"] = num_frames_tensor
        model_args["ar"] = ar_tensor
        model_args["fps"] = fps_tensor

        # compute latent size
        input_size = (num_frames, *resolution)
        latent_size = vae.get_latent_size(input_size)

        # process prompt
        prompt_raw = [prompt_text]
        prompt_raw, _ = extract_json_from_prompts(prompt_raw)
        prompt_loops = process_prompts(prompt_raw, num_loop)
        video_clips = []

        # prepare mask strategy
        if mode == "Text2Image":
            mask_strategy = [None]
        elif mode == "Text2Video":
            if reference_image is not None:
                mask_strategy = ["0"]
            else:
                mask_strategy = [None]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # =========================
        # 2. Load reference images
        # =========================
        if mode == "Text2Image":
            refs_x = collect_references_batch([None], vae, resolution)
        elif mode == "Text2Video":
            if reference_image is not None:
                # save image to disk
                from PIL import Image

                im = Image.fromarray(reference_image)

                with NamedTemporaryFile(suffix=".jpg") as temp_file:
                    im.save(temp_file.name)
                    refs_x = collect_references_batch([temp_file.name], vae, resolution)
            else:
                refs_x = collect_references_batch([None], vae, resolution)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 4.3. long video generation
        for loop_i in range(num_loop):
            # 4.4 sample in hidden space
            batch_prompts = [prompt[loop_i] for prompt in prompt_loops]
            z = torch.randn(
                len(batch_prompts),
                vae.out_channels,
                *latent_size,
                device=device,
                dtype=dtype,
            )

            # 4.5. apply mask strategy
            masks = None

            # if cfg.reference_path is not None:
            if loop_i > 0:
                ref_x = vae.encode(video_clips[-1])
                for j, refs in enumerate(refs_x):
                    if refs is None:
                        refs_x[j] = [ref_x[j]]
                    else:
                        refs.append(ref_x[j])
                    if mask_strategy[j] is None:
                        mask_strategy[j] = ""
                    else:
                        mask_strategy[j] += ";"
                    mask_strategy[
                        j
                    ] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length}"

            masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)

            # 4.6. diffusion sampling
            # hack to update num_sampling_steps and cfg_scale
            scheduler_kwargs = config.scheduler.copy()
            scheduler_kwargs.pop("type")
            scheduler_kwargs["num_sampling_steps"] = sampling_steps
            scheduler_kwargs["cfg_scale"] = cfg_scale

            scheduler.__init__(**scheduler_kwargs)
            samples = scheduler.sample(
                stdit,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
                mask=masks,  # scheduler must support mask
            )
            samples = vae.decode(samples.to(dtype))
            video_clips.append(samples)

            # 4.7. save video
            if loop_i == num_loop - 1:
                video_clips_list = [video_clips[0][0]] + [
                    video_clips[i][0][:, config.condition_frame_length :]
                    for i in range(1, num_loop)
                ]
                video = torch.cat(video_clips_list, dim=1)
                current_datetime = datetime.datetime.now()
                timestamp = current_datetime.timestamp()
                save_path = os.path.join(args.output, f"output_{timestamp}")
                saved_path = save_sample(
                    video, save_path=save_path, fps=config.fps // config.frame_interval
                )
                return saved_path


@spaces.GPU(duration=200)
def run_image_inference(*args, **kwargs):
    return run_inference("Text2Image", *args, **kwargs)


@spaces.GPU(duration=200)
def run_video_inference(*args, **kwargs):
    return run_inference("Text2Video", *args, **kwargs)


def main():
    # create demo
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                  <div>
                    <img class="logo" src="https://lambdalabs.com/hubfs/logos/lambda-logo.svg" alt="Lambda Logo"
                        style="margin: auto; max-width: 7rem;">
                    <h1 style="font-weight: 900; font-size: 3rem;">
                      text2bricks
                    </h1>
                  </div>
                  <p style="margin-bottom: 10px; font-size: 94%">
                  Generate a stop motion brick animation from a text description, fine tuned by Lambda Labs.
                  </p>
                  <p style="margin-bottom: 10px; font-size: 94%">
                  Running on an NVIDIA A100.
                  </p>
                </div>
                """
                )

        with gr.Row():
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your video here",
                    lines=4,
                )
                length = gr.Radio(
                    choices=["2s", "4s", "8s", "16s"],
                    value="2s",
                    label="Video Length (only effective for video generation)",
                    info="8s may fail as Hugging Face ZeroGPU has the limitation of max 200 seconds inference time.",
                )
                with gr.Row():
                    seed = gr.Slider(
                        value=1024, minimum=1, maximum=2048, step=1, label="Seed"
                    )
                    cfg_scale = gr.Slider(
                        value=7.0,
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        label="CFG Scale",
                    )
                with gr.Row():
                    video_gen_button = gr.Button("Generate video")

            with gr.Column():
                output_video = gr.Video(label="Output Video", height="100%")

        with gr.Row():
            with gr.Column():
                ex = gr.Examples(
                    [
                        [
                            "A teenage boy in a red puffer vest stands beside a futuristic, silver DeLorean car with its gull-wing doors open, as lightning strikes a clock tower in the background.",
                            1024,
                            7,
                            "2s",
                        ],
                        [
                            "Jack Nicholson breaking through the door with a axe and shouting 'Here's Johnny!'",
                            1024,
                            7,
                            "2s",
                        ],
                        [
                            "Alien - Ellen Ripley, in a spacesuit, confronting the Alien in the spaceship's airlock.",
                            1024,
                            7,
                            "2s",
                        ],
                    ],
                    inputs=[prompt_text, seed, cfg_scale, length],
                    outputs=output_video,
                    cache_examples=True,
                    fn=run_video_inference,
                )

        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                  <p style="margin-bottom: 10px; font-size: 94%">
                  If you want to find out how we made this model read about it in <a href="https://wandb.ai/lambdalabs/lego/reports/Text2Bricks-Finetuning-OpenSora-with-1-000-GPU-Hours--Vmlldzo4MDE3MTky">this blog post</a>.
                  </p>
                  <p style="margin-bottom: 10px; font-size: 94%">
                  And if you want to train your own text2bricks variants, see our <a href="https://github.com/LambdaLabsML/Open-Sora/blob/lambda_bricks/TUTORIAL.md">customized fork</a>!
                  </p>
                </div>
                """
                )

        video_gen_button.click(
            fn=run_video_inference,
            inputs=[prompt_text, seed, cfg_scale, length],
            outputs=output_video,
        )

    # launch
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)


if __name__ == "__main__":
    main()
