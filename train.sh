#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=mirage.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=1000:00:00

cd /mnt/opr/levlevi/opr/video-generation-hbm/Open-Sora
# source op-2
conda activate op-2

OMP_NUM_THREADS=52 /playpen-storage/levlevi/anaconda3/envs/op-2/bin/torchrun --master_port=25679 --nproc_per_node 8  scripts/train.py \
    configs/opensora-v1-1/train/text2bricks-360p-4f.py \
    --data-path /mnt/mir/fan23j/Open-Sora/data/bball/output_filtered.csv \
    --ckpt-path /mnt/mir/fan23j/Open-Sora/pretrained/OpenSora-STDiT-v2-stage3/model.safetensors