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
import tqdm
from openai import OpenAI

from tools.caption.batch.utils import create_batch_input, submit_batch_input, check_batches
from tools.caption.utils import VideoTextDataset, PROMPTS


def main(args):
    check_batches(args.key, description=args.desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str)
    parser.add_argument("--desc", type=str, default=None)
    args = parser.parse_args()

    main(args)
