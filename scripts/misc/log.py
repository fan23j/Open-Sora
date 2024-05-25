import os
import re
import argparse
from datetime import datetime

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
    parser.add_argument('--wandb-exp-name', type=str, default="my_awesome_experiment",
                        help="Experiment name for wandb.")
    parser.add_argument('--ckpt', action='store_true', help="Flag to log each ckpt as individual runs")
    
    # Parse the arguments
    args = parser.parse_args()

    return args

def main():    
    args = parse_arguments()
    if not args.ckpt:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        wandb.init(project=args.wandb_project_name, entity=args.wandb_project_entity, name=args.wandb_exp_name + "_" + timestamp)

    # Open the file and read the lines
    with open(args.eval_path, 'r') as file:
        path_evals = file.readlines()
        path_evals = [path.strip() for path in path_evals]
    
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

        if not args.ckpt:
            run_log(path, epoch, global_step, args.fps)
        else:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            exp_name = path.split("/")[-2] + "_ep" + epoch + "_step" + global_step  + "_" + timestamp
            wandb.init(project=args.wandb_project_name, entity=args.wandb_project_entity, name=exp_name)
            run_log(path, epoch, global_step, args.fps)
            wandb.finish()

    if not args.ckpt:
        wandb.finish()

if __name__ == "__main__":
    main()
