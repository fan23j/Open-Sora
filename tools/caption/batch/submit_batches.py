import argparse
import base64
import csv
import itertools
import json
import os
import sys
import time
from io import BytesIO

import requests
from tqdm import tqdm
from openai import OpenAI

from tools.caption.batch.utils import create_batch_input, submit_batch_input, filter_submitted_batches
from tools.caption.utils import VideoTextDataset, PROMPTS


def main(args):
    batchinput_file_paths = []
    for batch_file in os.listdir(args.input):
        batchinput_file_path = os.path.join(args.input, batch_file)
        batchinput_file_paths.append(batchinput_file_path)

    batchinput_file_paths = filter_submitted_batches(batchinput_file_paths, args.key)

    print(batchinput_file_paths)

    for batchinput_file_path in tqdm(list(batchinput_file_paths)):
        time.sleep(60)
        batch, new = submit_batch_input(batchinput_file_path, args.key)
        if new:
            print(f"Submitted batch: {batch}")
        if batch.errors is not None:
            print(f"Errors: {batch.errors}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the dir of prepared batches")
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    main(args)
