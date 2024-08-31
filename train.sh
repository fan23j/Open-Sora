#!/bin/bash
#SBATCH -D /mnt/opr/levlevi/opr/video-generation-hbm/Open-Sora
#SBATCH --partition=a6000
#SBATCH --nodelist=mirage.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=50:00:00

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH=/mnt/opr/levlevi/opr/video-generation-hbm/Open-Sora:$PYTHONPATH

cd /mnt/opr/levlevi/opr/video-generation-hbm/Open-Sora
/playpen-storage/levlevi/anaconda3/condabin/conda  activate op-2

RANDOM_PORT=$((20000 + RANDOM % 20000))
OMP_NUM_THREADS=52 /playpen-storage/levlevi/anaconda3/envs/op-2/bin/torchrun --master_port=$RANDOM_PORT --nproc_per_node 1  scripts/train.py \
    configs/opensora-v1-1/train/text2bricks-360p-4f.py \
    --data-path /mnt/mir/fan23j/Open-Sora/data/bball/output_filtered.csv \
    --ckpt-path /mnt/mir/fan23j/Open-Sora/pretrained/OpenSora-STDiT-v2-stage3/model.safetensors