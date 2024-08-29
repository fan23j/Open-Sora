# export PATH=/home/fan23j/anaconda3/envs/opensora/bin:$PATH
# TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=52 colossalai run --nproc_per_node 1 \

torchrun --master_port=25678 scripts/train.py \
    configs/opensora-v1-1/train/text2bricks-360p-64f.py \
    --data-path /mnt/mir/fan23j/Open-Sora/data/bball/output_filtered.csv \
    --ckpt-path /mnt/mir/fan23j/Open-Sora/pretrained/OpenSora-STDiT-v2-stage3/model.safetensors