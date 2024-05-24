import argparse
import base64
import csv
import itertools
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import tqdm
from openai import OpenAI

from tools.caption.batch.utils import create_batch_input
from tools.caption.utils import VideoTextDataset, PROMPTS
from tools.datasets.utils import extract_frames


def extract_images(dataset: VideoTextDataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = []
    for sample in dataset:
        video_path = Path(sample['path'])
        output_image_path = output_dir / (video_path.stem + '.jpg')
        try:
            image = extract_frames(str(video_path), points=[0.5], backend="opencv")[0]
            image.save(output_image_path)
        except Exception as e:
            print(e)
            continue
        sample = {'path': str(output_image_path), 'image': [image]}
        ds.append(sample)
    print(f'Extracted {len(ds)} frames')
    return ds

def main(args):
    dataset = VideoTextDataset(args.input)
    prompt = PROMPTS[args.prompt]["text"]

    if args.to_images:
        if args.image_output_dir is None:
            raise ValueError("Please specify the image output directory")
        dataset = extract_images(dataset, args.image_output_dir)

    create_batch_input(dataset, prompt, detail='low', max_size_mb=50, output_dir=args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--prompt", type=str, default="video-f3-detail-3ex")
    parser.add_argument("--output", type=str, default="batches")
    parser.add_argument("--to-images", default=False, action="store_true")
    parser.add_argument("--image-output-dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
