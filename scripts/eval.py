import time
import numpy as np
import os
import random
import re

import torch
import torch.distributed as dist
import colossalai
from colossalai.cluster import DistCoordinator
from tqdm import tqdm

from opensora.acceleration.parallel_states import (
    set_sequence_parallel_group,
)
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import (
    create_experiment_workspace,
    parse_configs,
)
from opensora.utils.misc import to_torch_dtype

from mmengine.runner import set_random_seed

def save_prompt(prompt, save_path):
    with open(save_path, "w") as file:
        file.write(prompt)    

def save_rng_state():
    rng_state = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }
    return rng_state

def set_seed_custom(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def ensure_parent_directory_exists(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")

z_log = None
def write_sample(model, text_encoder, vae, scheduler, cfg, output_path, dtype, device):
    prompts = cfg.prompt[dist.get_rank()::dist.get_world_size()]
    if prompts:
        global z_log   
        rng_state = save_rng_state()
        save_dir = os.path.join(
            cfg.exp_dir, output_path
        )
        with torch.no_grad():
            image_size = cfg.image_size
            num_frames = cfg.num_frames
            fps = cfg.fps
            batch_size = cfg.batch_size

            input_size = (num_frames, *image_size)
            latent_size = vae.get_latent_size(input_size)
            if z_log is None:
                rng = np.random.default_rng(seed=cfg.seed)
                z_log = rng.normal(size=(len(prompts), vae.out_channels, *latent_size))
            z = torch.tensor(z_log, device=device, dtype=float).to(dtype=dtype)
            set_seed_custom(cfg.seed)

            samples = []
            samples_prompt = []

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_z = z[i:i + batch_size]
                batch_samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=batch_z,
                    prompts=batch_prompts,
                    device=device,
                    additional_args=dict(
                        height=torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        width=torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        num_frames=torch.tensor([num_frames], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        ar=torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        fps=torch.tensor([fps], device=device, dtype=dtype).repeat(len(batch_prompts)),
                    ),
                )

                batch_samples = vae.decode(batch_samples.to(dtype))
                samples.extend(batch_samples)
                samples_prompt.extend(batch_prompts)

            # 4.4. save samples
            # if coordinator.is_master():
            for prompt, (sample_idx, sample) in zip(samples_prompt, enumerate(samples)):
                id = sample_idx * dist.get_world_size() + dist.get_rank()
                save_path = os.path.join(
                    save_dir, f"sample_{id}"
                )
                prompt_path = os.path.join(
                    save_dir, f"sample_{id}.txt"
                )
                ensure_parent_directory_exists(save_path)
                save_sample(
                    sample,
                    fps=fps,
                    save_path=save_path,
                )
                save_prompt(
                    prompt,
                    save_path=prompt_path
                )


def is_file_complete(file_path, interval=1, timeout=60):
    previous_size = -1
    elapsed_time = 0
    
    while elapsed_time < timeout:
        if os.path.isfile(file_path):
            current_size = os.path.getsize(file_path)
            if current_size == previous_size:
                return True  # File size hasn't changed, assuming file is complete
            previous_size = current_size
        
        time.sleep(interval)
        elapsed_time += interval
    
    return False


def run_eval(cfg, output_path, coordinator, device):
    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    dtype = to_torch_dtype(cfg.dtype)

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=cfg.enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # Run inference
    write_sample(model, text_encoder, vae, scheduler, cfg, output_path, dtype, device)


def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=False)
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False

    cfg.enable_sequence_parallelism = enable_sequence_parallelism

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    prompts = cfg.prompt

    # Open the file and read the lines
    with open(cfg.list_ckpts, 'r') as file:
        path_ckpts = file.readlines()
        path_ckpts = [path.strip() for path in path_ckpts]
    
    for path in path_ckpts:
        cfg.model["from_pretrained"] = path
        output_path = path.split("/")[-1]
        print(output_path)
        run_eval(cfg, output_path, coordinator, device)


if __name__ == "__main__":
    main()
