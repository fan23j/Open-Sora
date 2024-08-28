import os

# Set the environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "52"

# Activate the conda environment (assuming it's set up correctly)
os.environ["PATH"] = "/playpen-storage/levlevi/anaconda3/envs/op-2/bin:" + os.environ["PATH"]

# Construct the command
command = [
    "colossalai",
    "run",
    "--nproc_per_node",
    "1",
    "scripts/train.py",
]

# Execute the command within the same process
os.execvp(command[0], command)