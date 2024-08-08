#!/bin/bash
#SBATCH --job-name=cleaned_slurm  # Job name
#SBATCH --output=/mnt/mir/fan23j/Open-Sora/slurm_logs/%j.out       # Output log file (%x: job name, %j: job ID)
#SBATCH --error=/mnt/mir/fan23j/Open-Sora/slurm_logs/%j.err        # Error log file
#SBATCH --ntasks=4                    # Number of tasks (processes)
#SBATCH --gpus-per-task=1             # Number of GPUs per task
#SBATCH --cpus-per-task=52            # Number of CPU cores per task
#SBATCH --partition=h100    # Partition name (change to your partition)
#SBATCH --nodes=1                     # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=4           # Number of tasks per node
#SBATCH --mem=128G                    # Memory per node

source /home/fan23j/anaconda3/etc/profile.d/conda.sh
conda init
conda activate opensora

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=52

# Run the training script
srun ./scripts/train.py \
    configs/opensora-v1-1/train/text2bricks-360p-64f.py \
    --data-path  data/bball/output.csv \
    --ckpt-path pretrained/OpenSora-STDiT-v2-stage3/model.safetensors