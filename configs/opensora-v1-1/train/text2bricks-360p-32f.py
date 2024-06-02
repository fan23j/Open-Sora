# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=2,
    image_size=(None, None),
    transform_name="resize_crop",
)
bucket_config = {
    "360p": {32: (1.0, 8)},
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
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
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

# Others
seed = 1024
outputs = "outputs"
wandb = True

epochs = 1000
log_every = 10
ckpt_every = 250
load = None

lr_schedule = "1cycle"
anneal_strategy = "cos"
warmup_steps = 1500
cooldown_steps = 2500
lr = 1e-5
min_lr = 1e-7
max_lr=1e-4

batch_size = None
grad_clip = 1.0

eval_prompts = [
        "A scuba diver on a coral reef with schools of fish swimming, and a sea turtle and an octopus.",
        "People eating ice cream and drinkin espresso outside of a cafe on a narrow street in Rome. There are stores along the street selling a variety of wares. One shop sells fruits. Another shop sells vegetables. A third shop sells christmas ornaments. Many people walk along the street.",
        "An astronaut walking on the moon, with the effects of gravity making the walk appear very bouncy.",
        "A person walks down a garden path. The path is surrounded by gorgeous and colorful flowers, lush bushes, and grand trees. Butterflies and bees zip around the scene in the background. The person is walking directly towards the camera.",
        "A ninja wearing a red outfit jumps from one roof of a building to a second building's roof. The full moon is in sight directly behind the ninja.",
        "A newly married couple do their first dance at a wedding. Both partners are wearing white dresses, and are slow dancing in the center of a beautifully decorated wedding hall.",
        "A fat rabbit wearing a purple robe walking through a fantasy landscape",
        "A young man walks alone by the seaside",            
]

eval_image_size = (360, 640)
eval_num_frames = 32
eval_fps = 8
eval_batch_size = 1
eval_steps = ckpt_every

wandb_project_name = "text2bricks-360p-32f"
wandb_project_entity = None

exp_id = "text2bricks-360p-32f"
