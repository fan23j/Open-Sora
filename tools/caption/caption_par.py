import argparse
import base64
import csv
import os
from io import BytesIO

import requests
import tqdm
import multiprocessing as mp

from tools.caption.utils import PROMPTS, VideoTextDataset
from tools.datasets.utils import IMG_EXTENSIONS, VID_EXTENSIONS


MODELS = {
    "gpt4": "gpt-4-vision-preview",
    "gpt4o": "gpt-4o",
}

counter = None
counter_lock = None

def init(args_counter, args_counter_lock):
    global counter
    global counter_lock
    counter = args_counter
    counter_lock = args_counter_lock


def to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_caption(frame, prompt, api_key, model):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": MODELS[model],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame[2]}"}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    caption = response.json()["choices"][0]["message"]["content"]
    caption = caption.replace("\n", " ")
    return caption

def process_sample(myargs):
    sample, queue, num_samples, args = myargs
    prompt = PROMPTS[args.prompt]["text"]
    if "text" in args.prompt:
        prompt = prompt.format(sample["text"])
    frames = sample["image"]
    frames = [to_base64(frame) for frame in frames]
    try:
        caption = get_caption(frames, prompt, args.key, args.model)
        queue.put((sample["path"], caption))
    except Exception:
        queue.put((sample["path"], "Caption failed"))
    with counter_lock:
        counter.value += 1
    print("\r" + f"{counter.value}/{num_samples}", end='', flush=True)

def writer_process(queue, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "text"])
        while True:
            data = queue.get()
            if data == 'DONE':
                break
            writer.writerow(data)
    f.close()

def main(args):
    # ======================================================
    # 1. read video list
    # ======================================================
    dataset = VideoTextDataset(args.input)
    output_file = os.path.splitext(args.input)[0] + "_caption.csv"


    # make sure that the prompt type matches the data type
    data_extension = "." + dataset.data["path"].iloc[0].split(".")[-1]
    prompt_type = PROMPTS[args.prompt]["type"]
    if prompt_type == "image":
        assert (
            data_extension.lower() in IMG_EXTENSIONS
        ), "The prompt is suitable for an image dataset but the data is not image."
    elif prompt_type == "video":
        assert (
            data_extension.lower() in VID_EXTENSIONS
        ), "The prompt is suitable for a video dataset but the data is not video."
    else:
        raise ValueError(f"Found invalid prompt type {prompt_type}")

    manager = mp.Manager()
    counter = manager.Value('i', 0)
    counter_lock = mp.Lock()
    queue = manager.Queue()
    num_samples = len(dataset)

    with mp.Pool(processes=args.num_p, initializer=init, initargs=(counter, counter_lock)) as pool:
        pool.apply_async(writer_process, (queue, output_file))
        for _ in pool.imap_unordered(process_sample, [(sample, queue, num_samples, args) for sample in dataset]):
            pass

        queue.put('DONE')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--prompt", type=str, default="video-f3-detail-3ex")
    parser.add_argument("--key", type=str)
    parser.add_argument("--num-p", type=int, default=8)
    parser.add_argument("--model", type=str, choices=["gpt4", "gpt4o"], default="gpt4o")
    args = parser.parse_args()

    main(args)
