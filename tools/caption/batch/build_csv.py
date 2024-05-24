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

from tools.caption.batch.utils import create_batch_input, submit_batch_input, filter_submitted_batches, \
    download_batches, build_csv
from tools.caption.utils import VideoTextDataset, PROMPTS


def main(args):
    print(args.input, args.output)
    build_csv(args.input, args.output, args.type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--type", type=str, default="video")
    # parser.add_argument("--key", type=str)
    # parser.add_argument("--desc", type=str, default=None)
    args = parser.parse_args()

    main(args)