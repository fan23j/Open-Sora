import time
import numpy as np
import os
import random
import re
import argparse

import wandb

def run_log(path, epoch, global_step, fps):

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp4"):
                sample_idx = file.split("_")[-1]
                sample_path = os.path.join(path, file)
                prompt_path = os.path.join(path, ".".join(file.split(".")[:-1]) + ".txt")

                with open(prompt_path, "r") as prompt_file:
                    prompt = prompt_file.read().strip()
                    wandb.log(
                        {
                            "eval/prompt_" + sample_idx: wandb.Video(
                                sample_path,
                                caption=prompt,
                                format="mp4",
                                fps=fps,
                            )
                        },
                        step=int(global_step),
                    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for wandb configuration.")

    # Define the arguments
    parser.add_argument('--eval-path', type=str, default="",
                        help="Frames per second for wandb.")
    parser.add_argument('--fps', type=int, default=8,
                        help="Frames per second for wandb.")
    parser.add_argument('--wandb-project-name', type=str, default="brick_eval",
                        help="Name of the wandb project.")
    parser.add_argument('--wandb-project-entity', type=str, default="lambdalabs",
                        help="Entity name for the wandb project.")
    parser.add_argument('--wandb-exp-name', type=str, default="step1_24k_15k_64f_lowlr_maxbs",
                        help="Experiment name for wandb.")

    # Parse the arguments
    args = parser.parse_args()

    return args

def main():    
    args = parse_arguments()
    wandb.init(project=args.wandb_project_name, entity=args.wandb_project_entity, name=args.wandb_exp_name)

    # Open the file and read the lines
    with open(args.eval_path, 'r') as file:
        path_evals = file.readlines()
        path_evals = [path.strip() for path in path_evals]
    
    count = 0
    for path in path_evals:

        # Search for the pattern in the file path
        match_global_step = re.search(r'global_step(\d+)', path)
        match_epoch = re.search(r'epoch(\d+)', path)

        if match_global_step:
            global_step = match_global_step.group(1)
        else:
            print("No global_step found")
            sys.exit()
    
        if match_epoch:
            epoch = match_epoch.group(1)
        else:
            print("No epoch found")
            sys.exit()

        run_log(path, epoch, global_step, args.fps)

        count += 1
        if count == 2:
            break

if __name__ == "__main__":
    main()
