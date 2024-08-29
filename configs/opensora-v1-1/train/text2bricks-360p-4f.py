# define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=2,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {
    "360p": {4: (1.0, 2)},
}
mask_ratios = {
    "mask_no": 0.75,
    "mask_quarter_random": 0.025,
    "mask_quarter_head": 0.025,
    "mask_quarter_tail": 0.025,
    "mask_quarter_head_tail": 0.05,
    "mask_image_random": 0.025,
    "mask_image_head": 0.025,
    "mask_image_tail": 0.025,
    "mask_image_head_tail": 0.05,
}

# Define acceleration
num_workers = 8
num_bucket_build_workers = 8
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=False,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)
scheduler_inference = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)

# misc
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 500
load = None

lr_schedule = "cosine_const"
warmup_steps = 1
lr = 1e-5

batch_size = 1
grad_clip = 1.0

eval_prompts = [
    "A basketball player missing a three-point shot",
]
eval_image_size = (360, 640)
eval_num_frames = 4
eval_fps = 8
eval_batch_size = 1
eval_steps = ckpt_every

wandb_project_name = "STDiT-Motion"
wandb_project_entity = "Video-Generation-For-Structured-Behavior-Modeling"
exp_id = "multi_traj_unaligned"