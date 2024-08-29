# export PATH=/home/fan23j/anaconda3/envs/opensora/bin:$PATH
# TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=52 colossalai run --nproc_per_node 1 \

OMP_NUM_THREADS=52 torchrun --master_port=25679 --nproc_per_node 8 scripts/train.py \
    configs/opensora-v1-1/train/text2bricks-360p-4f.py \
    --data-path /mnt/mir/fan23j/Open-Sora/data/bball/output_filtered.csv \
    --ckpt-path /mnt/mir/fan23j/Open-Sora/pretrained/OpenSora-STDiT-v2-stage3/model.safetensors