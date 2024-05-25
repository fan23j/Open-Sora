import os
import sys
import argparse
import math

import wandb
import wandb.apis.reports as wr

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for configuration.")

    # Define the arguments
    parser.add_argument('--num-cols', type=int, default=2,
                        help="Number of columns in the grid panel.")
    parser.add_argument('--num-prompt', type=int, default=16,
                        help="Number of total prompt for wandb.")
    parser.add_argument('--wandb-project-name', type=str, default="brick_eval",
                        help="Project name on wandb.")
    parser.add_argument('--wandb-project-entity', type=str, default="lambdalabs",
                        help="Project entity on wandb.")
    parser.add_argument('--wandb-report-title', type=str, default="my_awesome_report",
                        help="Report title on wandb.")
    parser.add_argument(
        '--wandb-run-list',
        nargs='+',
        help='List of runs'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=["prompt"],
        default="prompt",
        help="Style of panel, one for each prompt or one for each run"
    )
    # Parse the arguments
    args = parser.parse_args()

    return args


def create_panel4prompt(args, report):
    rows_per_panel = math.ceil(len(args.wandb_run_list) / args.num_cols) if args.wandb_run_list else 2
    idx_prompt = range(0, args.num_prompt)
    report.blocks = []
    for idx_prompt in range(args.num_prompt):
        media_keys = f"eval/prompt_{idx_prompt}.mp4"
        blocks = [
            wr.PanelGrid(
                panels=[
                    wr.MediaBrowser(
                        media_keys=media_keys,
                        num_columns=args.num_cols, 
                        layout={'w': 24, 'h': 12 * rows_per_panel}
                    ),
                ],
                runsets=[
                    wr.Runset(
                        entity=args.wandb_project_entity, 
                        project=args.wandb_project_name,
                        filters={'displayName': {'$in': args.wandb_run_list}} if args.wandb_run_list else None,) 
                ],
            )
        ]
        if idx_prompt == 0:
            report.blocks = blocks
        else:
            report.blocks = report.blocks + blocks

    return report

def main():    
    if os.environ["WANDB_REPORT_API_ENABLE_V2"]:
        print("WANDB_REPORT_API_ENABLE_V2 has to be empty for run filters to work")
        sys.exit()

    args = parse_arguments()

    report = wr.Report(
        project = args.wandb_project_name,
        title = args.wandb_report_title,
        entity = args.wandb_project_entity,
        description = "",
    )

    if args.style == "prompt":
        report = create_panel4prompt(args, report)
    else:
        "We haven't implemented other styles yet"
        sys.exit()

    report.save()

if __name__ == "__main__":
    main()