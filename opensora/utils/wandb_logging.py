import time
import os
import wandb
import numpy as np
import torch
import torch.distributed as dist
import torchvision

from yaml import Token
from typing import Dict, List, Optional
from nba.src.entities.clip_annotations import ClipAnnotationWrapper
from opensora.models.stdit.stdit2 import STDiT2
from opensora.models.vae import VideoAutoencoderKL
from opensora.schedulers.iddpm import IDDPM
from opensora.utils.config_entity import TrainingConfig
from opensora.utils.rng import save_rng_state, set_seed_custom, load_rng_state
from opensora.datasets import (
    save_sample,
)


def ensure_parent_directory_exists(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


# TODO: figure out what the heck this func is doing
z_log = None


@torch.no_grad()
def write_sample(
    model: STDiT2,
    vae: VideoAutoencoderKL,
    scheduler: IDDPM,
    cfg: TrainingConfig,
    epoch: int,
    exp_dir: str,
    global_step: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Write an example eval to W&B.
    """

    # eval text prompts
    prompts = cfg.eval_prompts[dist.get_rank() :: dist.get_world_size()]
    if not prompts:
        return

    # latent
    global z_log

    rng_state = save_rng_state()
    back_to_train_model = model.training
    back_to_train_vae = vae.training

    # place models in inference mode
    vae = vae.eval()
    model = model.eval()

    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step + 1}")
    image_size = cfg.eval_image_size
    num_frames = cfg.eval_num_frames
    fps = cfg.eval_fps
    eval_batch_size = cfg.eval_batch_size

    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    if z_log is None:
        rng = np.random.default_rng(seed=42)
        z_log = rng.normal(size=(len(prompts), vae.out_channels, *latent_size))
    z = torch.tensor(z_log, device=device, dtype=float).to(dtype=dtype)
    set_seed_custom(42)

    samples = []
    conditions = torch.load("instance_trajs.pth")

    # transfer conditions to gpu
    for key in conditions:
        conditions[key] = conditions[key].to(device, dtype)

    for i in range(0, len(prompts), eval_batch_size):
        batch_prompts = prompts[i : i + eval_batch_size]
        batch_z = z[i : i + eval_batch_size]
        batch_samples = scheduler.sample(
            model,
            z=batch_z,
            prompts=batch_prompts,
            device=device,
            additional_args=dict(
                height=torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(
                    len(batch_prompts)
                ),
                width=torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(
                    len(batch_prompts)
                ),
                num_frames=torch.tensor(
                    [num_frames], device=device, dtype=dtype
                ).repeat(len(batch_prompts)),
                ar=torch.tensor(
                    [image_size[0] / image_size[1]], device=device, dtype=dtype
                ).repeat(len(batch_prompts)),
                fps=torch.tensor([fps], device=device, dtype=dtype).repeat(
                    len(batch_prompts)
                ),
                conditions=conditions,
            ),
        )
        batch_samples = vae.decode(batch_samples.to(dtype))
        samples.extend(batch_samples)

        # 4.4. save samples
        # if coordinator.is_master():
        for sample_idx, sample in enumerate(samples):
            id = sample_idx * dist.get_world_size() + dist.get_rank()
            save_path = os.path.join(save_dir, f"sample_{id}")
            ensure_parent_directory_exists(save_path)
            save_sample(
                sample,
                fps=fps,
                save_path=save_path,
            )

    if back_to_train_model:
        model = model.train()
    if back_to_train_vae:
        vae = vae.train()

    load_rng_state(rng_state)


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


def log_sample(
    is_master,
    cfg,
    epoch,
    exp_dir,
    global_step,
    check_interval=1,
    size_stable_interval=1,
):
    """
    Upload eval samples to W&B.
    """

    if not cfg.wandb:
        return

    for sample_idx, prompt in enumerate(cfg.eval_prompts):
        save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step + 1}")
        save_path = os.path.join(save_dir, f"sample_{sample_idx}")
        file_path = os.path.abspath(save_path + ".mp4")

        # TODO: dangerous, can get stuck here permanently if the file is not created
        while not os.path.isfile(file_path):
            time.sleep(check_interval)

        # File exists, now check if it is complete
        if is_file_complete(file_path, interval=size_stable_interval):
            if is_master:
                wandb.log(
                    {
                        f"eval/prompt_{sample_idx}": wandb.Video(
                            file_path,
                            caption=prompt,
                            format="mp4",
                            fps=cfg.eval_fps,
                        )
                    },
                    step=global_step,
                )
                print(f"{file_path} logged")
        else:
            print(f"{file_path} not found, skip logging.")


def save_training_samples_to_wnb(
    batch: Dict,
    is_master: bool,
    cfg: TrainingConfig,
    epoch: int,
    exp_dir: str,
    global_step: int,
    check_interval=1,
    size_stable_interval=1,
):
    """
    Create and save visualizations of training samples and conditions. Upload to W&B.
    """

    if not cfg.wandb or not is_master:
        return
    
    # temp out path
    dst_path = f"epoch{epoch}-global_step{global_step + 1}.mp4"
    
    # x: BCTWH
    # CTWH -> THWC
    # select the first video in the current training batch
    video_tensor: torch.Tensor = batch["unnormalized_video"][0, :, :, :, :].permute(0, 2, 3, 1)
    torchvision.io.write_video(dst_path, video_tensor, fps=cfg.eval_fps)

    # wait until video is done writing, log sample
    if is_file_complete(dst_path, interval=size_stable_interval):
        wandb.log(
            {
                f"eval/step_{global_step}": wandb.Video(
                    dst_path,
                    caption=f"Sample taken from step: {global_step}",
                    format="mp4",
                    fps=cfg.eval_fps,
                )
            },
            step=global_step,
        )

    # clean up
    os.remove(dst_path)
