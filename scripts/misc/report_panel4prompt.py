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
    # Parse the arguments
    args = parser.parse_args()

    return args

def main():    
    args = parse_arguments()
    report = wr.Report(
        project = args.wandb_project_name,
        title = args.wandb_report_title,
        entity = args.wandb_project_entity,
        description = "",
    )

    idx_prompt = range(0, args.num_prompt)
    num_runs = len(args.wandb_run_list)

    report.blocks = []
    rows_per_panel = math.ceil(num_runs / args.num_cols)

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
                        filters={'displayName': {'$in': args.wandb_run_list}},)
                ],
            )
        ]
        if idx_prompt == 0:
            report.blocks = blocks
        else:
            report.blocks = report.blocks + blocks
    report.save()

 
if __name__ == "__main__":
    main()