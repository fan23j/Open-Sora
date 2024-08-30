import warnings

from nba.src.entities.clip_dataset import FilteredClipDataset

# surpress warnings
warnings.filterwarnings("ignore")

import time
import wandb
import torch
import logging
import torch.distributed as dist
import numpy as np

from copy import deepcopy
from datetime import timedelta
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union, Any
from torch.utils.tensorboard import SummaryWriter

from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed

from nba.src.entities.clip_annotations import ClipAnnotationWrapper

# TODO: use STDiT3 as DiT backbone?
# https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3
from opensora.models.stdit.stdit2 import STDiT2
from opensora.models.vae import VideoAutoencoderKL
from opensora.schedulers.iddpm import IDDPM
from opensora.utils.config_entity import TrainingConfig
from opensora.utils.model_helpers import calculate_weight_norm, push_to_device
from opensora.utils.lr_schedulers import ConstantWarmupLR, OneCycleScheduler
from opensora.utils.wandb_logging import write_sample, log_sample
from opensora.utils.sampler_entities import MicroBatch
from opensora.datasets.datasets import NBAClipsDataset
from opensora.datasets.sampler import VariableNBAClipsBatchSampler
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import (
    prepare_variable_dataloader,
)
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import (
    create_logger,
    load,
    model_sharding,
    record_model_param_shape,
    save,
)
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import (
    all_reduce_mean,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, update_ema, log_progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)

CAPTION_CHANNELS = 4096
MODEL_MAX_LENGTH = 200


def process_batch(
    batch: Dict,
    vae: VideoAutoencoderKL,
    model: STDiT2,
    scheduler: IDDPM,
    mask_generator: MaskGenerator,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    :params:
    :batch: Dict = {
        "video": video,
        "text": text,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "ar": ar,
        "fps": video_fps,
        "conditions": conditions,
    }
    """

    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
    # y = batch.pop("text")

    # calculate visual and text encoding
    with torch.no_grad():
        model_args = dict()
        x = vae.encode(x)  # [B, C, T, H/P, W/P]

    # generate masks
    mask = mask_generator.get_masks(x)
    model_args["x_mask"] = mask

    # process additional batch items
    for k, v in batch.items():
        model_args[k] = push_to_device(v, device=device, dtype=dtype)

    # diffusion step
    t = torch.randint(
        low=0,
        high=scheduler.num_timesteps,
        size=(x.shape[0],),
        device=device,
    )

    # predict noise: () and calculate loss
    # 10GB -> 35GB
    loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)
    # pprint(loss_dict)
    return loss_dict


def compute_and_apply_gradients(
    loss: torch.Tensor,
    booster: Booster,
    optimizer: HybridAdam,
    lr_scheduler=None,
    ema=None,
    model=None,
):

    # TODO: we will try using the standard `backward` func
    # loss.backward()
    booster.backward(loss=loss, optimizer=optimizer)
    optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if ema is not None and model is not None:
        update_ema(ema, model, optimizer=optimizer)


def save_checkpoint_if_needed(
    cfg: TrainingConfig,
    global_step: int,
    epoch: int,
    step: int,
    booster: Booster,
    model: STDiT2,
    ema: STDiT2,
    optimizer: HybridAdam,
    lr_scheduler: IDDPM,
    coordinator: DistCoordinator,
    exp_dir: str,
    ema_shape_dict: dict,
    sampler_to_io,
):
    if cfg.ckpt_every > 0 and global_step % cfg.ckpt_every == 0 and global_step != 0:
        save(
            booster,
            model,
            ema,
            optimizer,
            lr_scheduler,
            epoch,
            step + 1,
            global_step + 1,
            cfg.batch_size,
            coordinator,
            exp_dir,
            ema_shape_dict,
            sampler=sampler_to_io,
        )
        logging.info(
            f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
        )


def finalize_epoch(
    cfg: TrainingConfig, dataloader: torch.utils.data.DataLoader, epoch: int
):
    if cfg.dataset.type == "VariableVideoTextDataset":
        dataloader.batch_sampler.set_epoch(epoch + 1)
        logging.info("Epoch done, recomputing batch sampler")


def train(
    cfg: TrainingConfig,
    coordinator: DistCoordinator,
    logger: logging.Logger,
    vae: VideoAutoencoderKL,
    model: STDiT2,
    scheduler: IDDPM,
    start_epoch: int,
    dataloader: torch.utils.data.DataLoader,
    mask_generator: MaskGenerator,
    num_steps_per_epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    booster: Booster,
    optimizer: HybridAdam,
    lr_scheduler: OneCycleScheduler,
    ema: STDiT2,
    writer: SummaryWriter,
    exp_dir: str,
    ema_shape_dict: Dict[str, Any],
    sampler_to_io: VariableNBAClipsBatchSampler,
    scheduler_inference: IDDPM,
) -> None:
    """
    Main training loop.
    """

    filtered_clip_dataset: FilteredClipDataset = sampler_to_io.dataset.filtered_dataset
    running_loss = 0.0
    log_step = 0
    acc_step = 0

    for epoch in range(start_epoch, cfg.epochs):
        dataloader_iter = iter(dataloader)

        # TODO: we can't see logging atm
        logger.info(f"Beginning epoch {epoch}...")

        pbar = tqdm(
            iterable=enumerate(dataloader_iter, start=0),
            desc=f"Training JAMES ⛹️ | Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
        )

        iteration_times = []
        for step, batch in pbar:
            start_time = time.time()
            
            # this discusting logic loads the annotation wrapper for the current sample
            sample_annotation_wrapper = ClipAnnotationWrapper(
                filtered_clip_dataset.filtered_clip_annotations_file_paths[
                    batch["clip_annotation_idx"].cpu().numpy()[0]
                ]
            )
            
            # HACK:
            # remove this addtional key to play nice with other models
            del batch["clip_annotation_idx"]

            # mem: 8.6s GB
            # process the batch
            loss_dict = process_batch(
                batch, vae, model, scheduler, mask_generator, device, dtype
            )
            loss = loss_dict["loss"].mean()

            # compute and apply gradients
            compute_and_apply_gradients(
                loss, booster, optimizer, lr_scheduler, ema, model
            )

            # logging
            global_step = epoch * num_steps_per_epoch + step
            iteration_times.append(time.time() - start_time)
            running_loss += loss.item()
            log_step += 1
            acc_step += 1
            
            running_loss, log_step = log_progress(
                logger,
                sample_annotation_wrapper,
                coordinator,
                global_step,
                cfg,
                writer,
                loss,
                running_loss,
                log_step,
                model,
                iteration_times,
                optimizer,
                epoch,
                acc_step,
            )

            # save checkpoint if needed
            save_checkpoint_if_needed(
                cfg,
                global_step,
                epoch,
                step,
                booster,
                model,
                ema,
                optimizer,
                lr_scheduler,
                coordinator,
                exp_dir,
                ema_shape_dict,
                sampler_to_io,
            )

            # evaluate and save samples if needed
            if global_step % cfg.eval_steps == 0:
                write_sample(
                    model,
                    vae,
                    scheduler_inference,
                    cfg,
                    epoch,
                    exp_dir,
                    global_step,
                    dtype,
                    device,
                )
                log_sample(coordinator.is_master(), cfg, epoch, exp_dir, global_step)

            pbar.update()

        # finalize the epoch
        finalize_epoch(cfg, dataloader, epoch)


def main():
    # parse command-line args
    args = parse_configs(training=True)
    cfg: TrainingConfig = TrainingConfig(args)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg.to_dict(), exp_dir)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)

    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    writer: Optional[Any] = None
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        logger = create_logger(exp_dir)
        logger.info("Training configuration:")
        pprint(cfg.to_dict())
        logger.info(f"Experiment directory created at {exp_dir}")
        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            PROJECT = cfg.wandb_project_name
            wandb.init(
                project=PROJECT,
                name=exp_name,
                config=cfg.to_dict(),
                # entity=cfg.wandb_project_entity,
            )

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = NBAClipsDataset(
        num_frames=cfg.dataset.num_frames,
        frame_interval=cfg.dataset.frame_interval,
        image_size=cfg.dataset.image_size,
        transform_name=cfg.dataset.transform_name,
    )
    logger.info(f"Dataset contains {len(dataset)} samples.")

    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )

    # TODO: use plugin's prepare dataloader
    dataloader = prepare_variable_dataloader(
        bucket_config=cfg.bucket_config,
        num_bucket_build_workers=cfg.num_bucket_build_workers,
        **dataloader_args,
    )

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model

    # the autoencoder, compresses videos by spatial and temp axis
    vae: VideoAutoencoderKL = build_module(cfg.vae.to_dict(), MODELS)

    # (None, None, None)
    input_size = (dataset.num_frames, *dataset.image_size)

    # [None, None, None]
    latent_size = vae.get_latent_size(input_size)

    # construct the  diffusion transformer
    # we use: STDiT2-XL/2
    model: STDiT2 = build_module(
        module=cfg.model.to_dict(),
        builder=MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=CAPTION_CHANNELS,
        model_max_length=MODEL_MAX_LENGTH,
    )
    # get parameter counts
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    # model that tracks exponential moving avg of params
    ema: STDiT2 = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict: Dict = record_model_param_shape(ema)

    # ~8.5 GB used at this stage
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler: IDDPM = build_module(cfg.scheduler.to_dict(), SCHEDULERS)
    scheduler_inference: IDDPM = build_module(
        cfg.scheduler_inference.to_dict(), SCHEDULERS
    )

    # 4.5. setup optimizer
    optimizer: HybridAdam = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    lr_scheduler: ConstantWarmupLR = ConstantWarmupLR(
        optimizer, factor=1, warmup_steps=cfg.warmup_steps, last_epoch=-1
    )

    # set grad checkpoint
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)

    logging.info("Begining Training")
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()

    # TODO: mask ratios are never `None`
    mask_generator = MaskGenerator(cfg.mask_ratios)

    # set default dtype
    torch.set_default_dtype(dtype)

    # boost model, optimizer, lr_scheduler, dataloader
    # TODO: boosting model raises mem usage 8.5 -> 14.1 GB
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )

    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")

    # TODO: we always use VariableVideoTextDataset
    assert type(dataloader.batch_sampler) is VariableNBAClipsBatchSampler
    num_steps_per_epoch = (
        dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    )

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = 0
    sampler_to_io = dataloader.batch_sampler
    assert type(sampler_to_io) is VariableNBAClipsBatchSampler
    logger.info(
        f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch"
    )

    # split ema
    model_sharding(ema)

    ## SAVE SAMPLE ##
    # first_global_step = start_epoch * num_steps_per_epoch + start_step
    # write_sample(
    #     model,
    #     vae,
    #     scheduler_inference,
    #     cfg,
    #     start_epoch,
    #     exp_dir,
    #     first_global_step,
    #     dtype,
    #     device,
    # )
    # log_sample(coordinator.is_master(), cfg, start_epoch, exp_dir, first_global_step)
    # logger.info("First global step done")

    # train the model
    train(
        cfg,
        coordinator,
        logger,
        vae,
        model,
        scheduler,
        start_epoch,
        dataloader,
        mask_generator,
        num_steps_per_epoch,
        device,
        dtype,
        booster,
        optimizer,
        lr_scheduler,
        ema,
        writer,
        exp_dir,
        ema_shape_dict,
        sampler_to_io,
        scheduler_inference,
    )


if __name__ == "__main__":
    main()
