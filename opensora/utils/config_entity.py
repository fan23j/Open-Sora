from typing import Dict, Optional, Tuple, Any, List


class DatasetConfig:
    def __init__(self, args: Dict) -> None:
        self.type: Optional[str] = args["type"]
        assert self.type in [
            "VariableVideoTextDataset"
        ], f"Error: expected dataset.type in {['VariableVideoTextDataset']}"
        self.data_path: Optional[str] = args["data_path"]
        self.num_frames: Optional[int] = args["num_frames"]
        self.frame_interval: int = args["frame_interval"]
        self.image_size: Tuple[Optional[int], Optional[int]] = args["image_size"]
        self.transform_name: Optional[str] = args["transform_name"]

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "data_path": self.data_path,
            "num_frames": self.num_frames,
            "frame_interval": self.frame_interval,
            "image_size": self.image_size,
            "transform_name": self.transform_name,
        }


class DiTModelConfig:
    def __init__(self, args: Dict) -> None:
        self.type: str = args["type"]
        self.from_pretrained: Optional[str] = args["from_pretrained"]
        self.input_sq_size: int = args["input_sq_size"]
        self.qk_norm: bool = args["qk_norm"]
        self.enable_flash_attn: bool = args["enable_flash_attn"]
        self.enable_layernorm_kernel: bool = args["enable_layernorm_kernel"]

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "from_pretrained": self.from_pretrained,
            "input_sq_size": self.input_sq_size,
            "qk_norm": self.qk_norm,
            "enable_flash_attn": self.enable_flash_attn,
            "enable_layernorm_kernel": self.enable_layernorm_kernel,
        }


class VAEConfig:
    def __init__(self, args: Dict) -> None:
        self.type: str = args["type"]
        self.from_pretrained: Optional[str] = args["from_pretrained"]
        self.micro_batch_size: int = args["micro_batch_size"]
        self.local_files_only: bool = args["local_files_only"]

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "from_pretrained": self.from_pretrained,
            "micro_batch_size": self.micro_batch_size,
            "local_files_only": self.local_files_only,
        }


class SchedulerConfig:
    def __init__(self, args: Dict) -> None:
        self.type: str = args["type"]
        self.timestep_respacing: str = args["timestep_respacing"]

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "timestep_respacing": self.timestep_respacing,
        }


class SchedulerInferenceConfig:
    def __init__(self, args: Dict) -> None:
        self.type: str = args["type"]
        self.num_sampling_steps: int = args["num_sampling_steps"]
        self.cfg_scale: float = args["cfg_scale"]
        self.cfg_channel: int = args["cfg_channel"]

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "num_sampling_steps": self.num_sampling_steps,
            "cfg_scale": self.cfg_scale,
            "cfg_channel": self.cfg_channel,
        }


class TrainingConfig:
    """
    Object containing all arguments required to execute a training run.
    """

    def __init__(self, args) -> None:
        self.dataset: DatasetConfig = DatasetConfig(args.dataset)
        self.bucket_config: Dict[str, Dict] = args.bucket_config
        assert (
            self.bucket_config is not None
        ), f"Error: `bucket_config` must not be `None`"
        self.num_workers: Optional[int] = args.num_workers
        self.num_bucket_build_workers: Optional[int] = args.num_bucket_build_workers

        self.dtype: str = args.dtype
        assert self.dtype in [
            "fp32",
            "fp16",
            "bf16",
        ], f"Error: invalid dtype, expected dtype in {['fp32', 'fp16', 'bf16']}"
        self.grad_checkpoint: bool = args.grad_checkpoint
        self.plugin: str = args.plugin
        self.sp_size: int = args.sp_size

        self.model: DiTModelConfig = DiTModelConfig(args.model)
        self.vae: VAEConfig = VAEConfig(args.vae)
        self.scheduler: SchedulerConfig = SchedulerConfig(args.scheduler)
        self.scheduler_inference: SchedulerInferenceConfig = SchedulerInferenceConfig(
            args.scheduler_inference
        )

        self.seed: int = args.seed
        self.outputs: str = args.outputs
        self.wandb: bool = args.wandb
        self.epochs: int = args.epochs
        self.log_every: int = args.log_every
        self.ckpt_every: int = args.ckpt_every
        self.load: Any = args.load

        self.lr_schedule: str = args.lr_schedule
        self.warmup_steps: int = args.warmup_steps
        self.lr: float = args.lr
        self.batch_size: Optional[int] = args.batch_size
        self.grad_clip: float = args.grad_clip

        self.eval_prompts: List[str] = args.eval_prompts
        self.eval_image_size: Tuple[int, int] = args.eval_image_size
        self.eval_fps: int = args.eval_fps
        self.eval_batch_size: Optional[int] = args.eval_batch_size
        self.eval_steps: int = args.eval_steps
        self.eval_num_frames: int = args.eval_num_frames

        self.wandb_project_name: str = args.wandb_project_name
        self.wandb_project_entity: str = args.wandb_project_entity
        self.exp_id: str = args.exp_id
        self.mask_ratios: Optional["Dict"] = args.mask_ratios

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset.to_dict(),
            "bucket_config": self.bucket_config,
            "num_workers": self.num_workers,
            "num_bucket_build_workers": self.num_bucket_build_workers,
            "dtype": self.dtype,
            "grad_checkpoint": self.grad_checkpoint,
            "plugin": self.plugin,
            "sp_size": self.sp_size,
            "model": self.model.to_dict(),
            "vae": self.vae.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "scheduler_inference": self.scheduler_inference.to_dict(),
            "seed": self.seed,
            "outputs": self.outputs,
            "wandb": self.wandb,
            "epochs": self.epochs,
            "log_every": self.log_every,
            "ckpt_every": self.ckpt_every,
            "load": self.load,
            "lr_schedule": self.lr_schedule,
            "warmup_steps": self.warmup_steps,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "grad_clip": self.grad_clip,
            "eval_prompts": self.eval_prompts,
            "eval_image_size": self.eval_image_size,
            "eval_fps": self.eval_fps,
            "eval_batch_size": self.eval_batch_size,
            "eval_steps": self.eval_steps,
            "eval_num_frames": self.eval_num_frames,
            "wandb_project_name": self.wandb_project_name,
            "wandb_project_entity": self.wandb_project_entity,
            "exp_id": self.exp_id,
        }
