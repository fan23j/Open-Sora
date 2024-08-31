from copy import deepcopy
from datetime import timedelta
from pprint import pprint
import time
import numpy as np
import os
import random

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader, save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema


from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from typing import List
class WarmupScheduler(_LRScheduler):
    """Starts with a log space warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to warmup lr in log space until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After warmup_epochs, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs: int, after_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.min_lr  = 1e-7
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                self.finished = True
            return self.after_scheduler.get_lr()

        # log linear
        #return [self.min_lr * ((lr / self.min_lr) ** ((self.last_epoch + 1) / self.warmup_epochs)) for lr in self.base_lrs]

        # cosine warmup
        return [self.min_lr + (lr - self.min_lr) * 0.5 * (1 - torch.cos(torch.tensor((self.last_epoch + 1) / self.warmup_epochs * torch.pi))) for lr in self.base_lrs]


    def step(self, epoch: int = None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)

class ConstantWarmupLR(WarmupScheduler):
    """Multistep learning rate scheduler with warmup.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        num_steps_per_epoch (int, optional): Number of steps per epoch, defaults to -1.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        factor: float,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        base_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=-1)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List

class OneCycleScheduler(_LRScheduler):
    """Implements the 1-cycle learning rate policy with warmup and cooldown.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle for each parameter group.
        total_steps (int): The total number of steps in the cycle.
        warmup_steps (int): Number of steps to warm up the learning rate.
        cooldown_steps (int): Number of steps to cool down the learning rate.
        final_lr (float): The final learning rate at the end of the cooldown.
        min_lr (float): The minimum learning rate to start with.
        anneal_strategy (str): {'cos', 'linear'} Learning rate annealing strategy.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_lr, warmup_steps, cooldown_steps, final_lr=0.001, min_lr=1e-7, anneal_strategy='cos', last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = 1e6
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.final_lr = final_lr
        self.min_lr = min_lr
        self.anneal_strategy = anneal_strategy

        self.step_size_up = self.warmup_steps
        self.step_size_down = self.cooldown_steps

        self.anneal_func = self._cosine_annealing if self.anneal_strategy == 'cos' else self._linear_annealing

        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)

    def _cosine_annealing(self, step, start_lr, end_lr, step_size):
        cos_out = torch.cos(torch.tensor(math.pi * step / step_size)) + 1
        return end_lr + (start_lr - end_lr) / 2.0 * cos_out

    def _linear_annealing(self, step, start_lr, end_lr, step_size):
        return end_lr + (start_lr - end_lr) * (step / step_size)

    def get_lr(self):
        if self.last_epoch < self.step_size_up:
            # Warm-up phase
            lr = [self.anneal_func(self.last_epoch, self.min_lr, self.max_lr, self.step_size_up) for _ in self.base_lrs]
        elif self.last_epoch < self.step_size_up + self.step_size_down:
            # Cooldown phase
            step = self.last_epoch - self.step_size_up
            lr = [self.anneal_func(step, self.max_lr, self.final_lr, self.step_size_down) for _ in self.base_lrs]
        else:
            # Constant phase
            lr = [self.final_lr for _ in self.base_lrs]
        return lr

    def step(self, epoch=None):
        if self.last_epoch == -1:
            if epoch is None:
                self.last_epoch = 0
            else:
                self.last_epoch = epoch
        else:
            self.last_epoch = epoch if epoch is not None else self.last_epoch + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def save_rng_state():
    rng_state = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }
    return rng_state


def load_rng_state(rng_state):
    torch.set_rng_state(rng_state['torch'])
    if rng_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
    np.random.set_state(rng_state['numpy'])
    random.setstate(rng_state['random'])


#from mmengine.runner import set_random_seed
def set_seed_custom(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #set_random_seed(seed=seed)


def calculate_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def ensure_parent_directory_exists(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


z_log = None
def write_sample(model, vae, scheduler, cfg, epoch, exp_dir, global_step, dtype, device):
    prompts = cfg.eval_prompts[dist.get_rank()::dist.get_world_size()]
    if prompts:
        global z_log   
        rng_state = save_rng_state()
        back_to_train_model = model.training
        back_to_train_vae = vae.training
        vae = vae.eval()
        model = model.eval()

        save_dir = os.path.join(
            exp_dir, f"epoch{epoch}-global_step{global_step + 1}"
        )

        with torch.no_grad():
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

            conditions = torch.load('/mnt/mir/fan23j/data/nba-plus-statvu-dataset/__scripts__/conditions.pth')

            #transfer conditions to gpu
            for key in conditions:
                conditions[key] = conditions[key].to(device, dtype)
            # reshape bbox ratios
            conditions["bbox_ratios"] = conditions["bbox_ratios"].permute(0, 2, 1, 3).contiguous()

            for i in range(0, len(prompts), eval_batch_size):
                batch_prompts = prompts[i:i + eval_batch_size]
                batch_z = z[i:i + eval_batch_size]
                batch_samples = scheduler.sample(
                    model,
                    z=batch_z,
                    prompts=batch_prompts,
                    device=device,
                    additional_args=dict(
                        height=torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        width=torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        num_frames=torch.tensor([num_frames], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        ar=torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        fps=torch.tensor([fps], device=device, dtype=dtype).repeat(len(batch_prompts)),
                        conditions=conditions,
                    ),
                )

                batch_samples = vae.decode(batch_samples.to(dtype))
                samples.extend(batch_samples)

            # 4.4. save samples
            # if coordinator.is_master():
            for sample_idx, sample in enumerate(samples):
                id = sample_idx * dist.get_world_size() + dist.get_rank()
                save_path = os.path.join(
                    save_dir, f"sample_{id}"
                )
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

def log_sample(is_master, cfg, epoch, exp_dir, global_step, check_interval=1, size_stable_interval=1):
    if cfg.wandb:
        for sample_idx, prompt in enumerate(cfg.eval_prompts):
            save_dir = os.path.join(
                exp_dir, f"epoch{epoch}-global_step{global_step + 1}"
            )
            save_path = os.path.join(
                save_dir, f"sample_{sample_idx}"
            )
            file_path = os.path.abspath(save_path + ".mp4")
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


def push_to_device(item, device, dtype):
    if isinstance(item, dict):
        return {k: push_to_device(v, device, dtype) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return type(item)(push_to_device(v, device, dtype) for v in item)
    elif hasattr(item, 'to'):
        return item.to(device, dtype)
    else:
        return item




def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        print(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            PROJECT=cfg.wandb_project_name
            wandb.init(project=PROJECT, entity=cfg.wandb_project_entity, name=exp_name, config=cfg._cfg_dict)

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
    dataset = build_module(cfg.dataset, DATASETS)
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
    if cfg.bucket_config is None:
        dataloader = prepare_dataloader(**dataloader_args)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.bucket_config,
            num_bucket_build_workers=cfg.num_bucket_build_workers,
            **dataloader_args,
        )
    if cfg.dataset.type == "VideoTextDataset":
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
        logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    vae = build_module(cfg.vae, MODELS)
    input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=4096,
        model_max_length=200,
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    scheduler_inference = build_module(cfg.scheduler_inference, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    if cfg.load is not None:
        lr_scheduler = None
    else:
        if cfg.lr_schedule == "cosine_const":
            lr_scheduler = ConstantWarmupLR(optimizer, factor=1, warmup_steps=cfg.warmup_steps, last_epoch=-1)
        elif cfg.lr_schedule == "1cycle":
            lr_scheduler = OneCycleScheduler(optimizer, min_lr=cfg.min_lr, max_lr=cfg.max_lr, final_lr=cfg.lr, warmup_steps=cfg.warmup_steps, cooldown_steps=cfg.cooldown_steps, anneal_strategy=cfg.anneal_strategy)
        else:
            lr_scheduler = None
    
    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    if cfg.dataset.type == "VariableVideoTextDataset":
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    else:
        num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    sampler_to_io = dataloader.batch_sampler if cfg.dataset.type == "VariableVideoTextDataset" else None
    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            model,
            ema,
            optimizer,
            None,# lr_scheduler,
            cfg.load,
            sampler=sampler_to_io if not cfg.start_from_scratch else None,
        )
        if not cfg.start_from_scratch:
            start_epoch, start_step, sampler_start_idx = ret
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")

        
        optim_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Overwriting loaded learning rate from {optim_lr} to config lr={cfg.lr}")
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    if cfg.dataset.type == "VideoTextDataset":
        dataloader.sampler.set_start_index(sampler_start_idx)

    model_sharding(ema)
    # log prompts for pre-training ckpt
    first_global_step = start_epoch * num_steps_per_epoch + start_step
    # write_sample(model, vae, scheduler_inference, cfg, start_epoch, exp_dir, first_global_step, dtype, device)
    # log_sample(coordinator.is_master(), cfg, start_epoch, exp_dir, first_global_step)
    print("First global step done")

    # 6.2. training loop
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            iteration_times = []
            for step, batch in pbar:
                start_time = time.time()
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")
                # Visual and text encoding
                with torch.no_grad():
                    model_args = dict()
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]

                # Mask
                if cfg.mask_ratios is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask
                else:
                    mask = None

                # Video info and conditions
                for k, v in batch.items():
                    model_args[k] = push_to_device(v, device, dtype)

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)

                # Backward & update
                loss = loss_dict["loss"].mean()
                #loss.backward(retain_graph=True)
                # def hook(grad):
                #     return grad
                # loss.register_hook(hook)
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1
                iteration_times.append(time.time() - start_time)


                # Log to tensorboard
                if coordinator.is_master() and global_step % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)

                    weight_norm = calculate_weight_norm(model)
                    #TODO: log training reconstruction and trajectory.
                    if cfg.wandb:
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
                        iteration_times = []

                # Save checkpoint
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
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

                    # log prompts for each checkpoints
                if global_step % cfg.eval_steps == 0:
                    write_sample(model, vae, scheduler_inference, cfg, epoch, exp_dir, global_step, dtype, device)
                    log_sample(coordinator.is_master(), cfg, epoch, exp_dir, global_step)

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
