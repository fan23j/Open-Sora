import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def check_status(command, description):
    try:
        subprocess.check_call(command, shell=True)
        print(f"Success: {description}")
    except subprocess.CalledProcessError:
        print(f"Error: {description} failed.", file=sys.stderr)
        sys.exit(1)

def download_videos(url_file, root_video):
    command = f"yt-dlp -f 'best[ext=mp4]' -o '{root_video}/%(title)s-%(id)s.%(ext)s' -a {url_file}"
    print(f"Downloading video from URL: {url_file}")
    check_status(command, f"download videos from {url_file}")

def main(args):
    root_dir = Path(args.output)
    root_dir.mkdir(parents=True, exist_ok=True)
    root_video = root_dir / 'video'
    root_clips = root_dir / 'clips'
    root_meta = root_dir / 'meta'

    # Create subdirectories
    root_video.mkdir(parents=True, exist_ok=True)
    root_clips.mkdir(parents=True, exist_ok=True)
    root_meta.mkdir(parents=True, exist_ok=True)

    if args.video_dir is not None:
        # Prepare videos
        command = f"cp -r {args.video_dir} {root_video}"
        check_status(command, "copy videos")

    if args.url_file is not None:
        # Download videos
        download_videos(args.url_file, root_video)

    # Continue with the rest of the operations
    commands = [
        (f"python -m tools.datasets.convert video {root_video} --output {root_meta / 'meta.csv'}", "convert video"),
        (f"python -m tools.datasets.datautil {root_meta / 'meta.csv'} --info --fmin 1", "datautil info"),
        (f"python -m tools.scene_cut.scene_detect {root_meta / 'meta_info_fmin1.csv'} --min-scene-len {args.min_scene_len}", "scene_detect"),
        (f"python -m tools.scene_cut.cut {root_meta / 'meta_info_fmin1_timestamp.csv'} --save_dir {root_clips}", "scene_cut cut"),
        (f"python -m tools.datasets.convert video {root_clips} --output {root_meta / 'meta_clips.csv'}", "convert clips"),
        (f"python -m tools.datasets.datautil {root_meta / 'meta_clips.csv'} --info --fmin 1", "datautil clips info")
    ]

    for command, description in commands:
        check_status(command, description)

    # Get the number of GPUs available
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True)
        num_gpus = int(result.strip())
        print(f"Number of GPUs available: {num_gpus}")
    except subprocess.CalledProcessError:
        print("Error: Failed to get the number of GPUs.", file=sys.stderr)
        sys.exit(1)

    # Aesthetic inference
    command = f"torchrun --nproc_per_node {num_gpus} -m tools.scoring.aesthetic.inference {root_meta / 'meta_clips_info_fmin1.csv'} --bs 16 --num_workers 8"
    check_status(command, "aesthetic inference")

    # Merge CSVs
    csvs_to_concat = glob.glob(str(root_meta / 'meta_clips_info_fmin1_aes_part*.csv'))
    command = f"python -m tools.datasets.datautil {' '.join(csvs_to_concat)} --output {root_meta / 'meta_clips_info_fmin1_aes.csv'}"
    check_status(command, "datautil aes merge")

    # Filter by aesthetic score
    command = f"python -m tools.datasets.datautil {root_meta / 'meta_clips_info_fmin1_aes.csv'} --aesmin {args.aes_score}"
    check_status(command, "datautil aesmin")

    if args.caption is None:
        return

    # Generate captions
    if args.caption == 'gpt4':
        command = f"python -m tools.caption.caption_par {root_meta / f'meta_clips_info_fmin1_aes_aesmin{args.aes_score}.csv'} --prompt {args.prompt} --key {args.key} --model {args.caption} --num-p {args.num_p}"
        check_status(command, "caption_gpt4")
    elif args.caption == 'gpt4o':
        command = f"python -m tools.caption.caption_par {root_meta / f'meta_clips_info_fmin1_aes_aesmin{args.aes_score}.csv'} --prompt {args.prompt} --key {args.key} --model {args.caption} --num-p {args.num_p}"
        check_status(command, "caption_gpt4o")

    command = f"python -m tools.datasets.datautil {root_meta / f'meta_clips_info_fmin1_aes_aesmin{args.aes_score}_caption.csv'} --video-info --clean-caption --refine-llm-caption --remove-empty-caption --output {root_meta / f'meta_clips_caption_cleaned{args.aes_score}.csv'}"
    check_status(command, "datautil clean-caption")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos from a list of URLs.')
    parser.add_argument('--video-dir', type=str, help='Path to the directory of prepared videos.')
    parser.add_argument('--url-file', type=str, help='Path to the text file containing video URLs.')
    parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)
    parser.add_argument('--caption', choices=['gpt4', 'gpt4o'], default='gpt4o', help='Captioning model to use.')
    parser.add_argument('--prompt', type=str, default='video-f3-detail-3ex', help='Prompt to use for captioning.')
    parser.add_argument('--key', type=str, help='OpenAI API key.')
    parser.add_argument("--num-p", type=int, default=8, help="Number of parallelized OpenAI API requests")
    parser.add_argument("--aes-score", type=float, default=5.0, help="Minimum aesthetic score for keeping a sample")
    parser.add_argument("--min-scene-len", type=int, default=90, help="Minimal number of frames for scenedetect")
    args = parser.parse_args()

    if args.video_dir is None and args.url_file is None:
        print("Error: Please specify either --video-dir or --url-file.", file=sys.stderr)
        sys.exit(1)

    if args.caption is not None and args.key is None:
        print("Error: Please specify the OpenAI API key with --key.", file=sys.stderr)
        sys.exit(1)


    main(args)
