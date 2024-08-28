import base64
import collections
import csv
import datetime
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import openai
from openai import OpenAI
from openai._legacy_response import LegacyAPIResponse
from openai.pagination import SyncCursorPage
from openai.types import Batch
from tqdm import tqdm

# Function to sanitize the filename
def sanitize_filename(filename):
    # Remove any characters that are not alphanumeric, dashes, underscores, or dots
    return re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)

def to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def create_openai_request(sample, prompt, detail='auto'):
    frames = [to_base64(frame) for frame in sample["image"]]
    img_messages = [
        {"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{frame}",
            "detail": detail
        }} for frame in frames
    ]
    openai_request = {
        "custom_id": f"{sample['path']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        *img_messages,
                    ],
                }
            ],
            "max_tokens": 300,
        }
    }
    return openai_request

def create_batch_input(dataset, prompt, detail='auto', max_size_mb=50, output_dir='batches', name=''):
    timestamp = sanitize_filename(datetime.datetime.now().isoformat()[0:19])
    batch_input_dir = os.path.join(output_dir, name + timestamp)
    os.makedirs(batch_input_dir, exist_ok=True)
    start = 0
    batchinput_file_path = os.path.join(batch_input_dir, f'batchinput-{name}{timestamp}-{start:03d}.jsonl')

    file = open(batchinput_file_path, 'w')
    with tqdm(total=len(dataset)) as pbar:
        for i, sample in enumerate(dataset):
            # If size in MB > max_size_mb, create a new file\
            size = os.path.getsize(batchinput_file_path)
            if size > max_size_mb * 1024 * 1024:
                file.close()
                start = i
                batchinput_file_path = os.path.join(batch_input_dir, f'batchinput-{name}{timestamp}-{start:03d}.jsonl')
                file = open(batchinput_file_path, 'w')

            openai_request = create_openai_request(sample, prompt, detail)
            file.write(json.dumps(openai_request) + '\n')
            sample_tag = '-'.join(sample["path"].split("-")[-2:])
            custom_info = {"Batch": f'{start:03d}', "Size": f'{size / 1024 / 1024:8.3f}MB', "Sample": sample_tag}
            pbar.update(1)
            pbar.set_postfix(custom_info)


        file.close()

def filter_submitted_batches(batchinput_file_paths, api_key):
    client = OpenAI(api_key=api_key)
    unsubmitted_batches = set(batchinput_file_paths)
    batches = []
    for batch in client.batches.list():
        batches.append(batch)
        time.sleep(1)
    for batchinput_file_path in batchinput_file_paths:
        description = Path(batchinput_file_path).stem
        for batch in batches:
            if description in batch.metadata['description']:
                unsubmitted_batches.remove(batchinput_file_path)
    return unsubmitted_batches

def submit_batch_input(batchinput_file_path, api_key):
    client = OpenAI(api_key=api_key)
    description = Path(batchinput_file_path).stem

    # batches = client.batches.list()
    # for batch in batches:
    #     if description in batch.metadata['description']:
    #         return batch, False

    batch_input_file = client.files.create(
        file=open(batchinput_file_path, "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description,
        }
    )
    return batch, True

def check_batches(api_key, description=None):
    counter = collections.Counter()
    client = OpenAI(api_key=api_key)
    batches = []

    # response: LegacyAPIResponse[SyncCursorPage[Batch]] = client.batches.with_raw_response.list()
    # print(response.headers)
    # # batch = response.parse()
    # # print(batch)
    # return


    for batch in client.batches.list():
    # for batch in tqdm(client.batches.list()):
        # print(type(response))
        # batch = response.parse()
        # print(batch)
        # return
        # time.sleep(0.03)
        batches.append(batch)

    for batch in batches:
        # time.sleep(1)
        if description is not None and description not in batch.metadata['description']:
            continue
        print(f"{batch.created_at} {batch.id} {batch.status} {batch.metadata['description']}")
        counter[batch.status] += 1
        if batch.errors:
            print(f"Errors: {batch.errors}")
    # print(counter)
    return batches

def cancel_batches(api_key, description=None):
    client = OpenAI(api_key=api_key)
    batches = check_batches(api_key, description=description)
    for batch in batches:
        if description is not None and description not in batch.metadata['description']:
            continue
        try:
            client.batches.cancel(batch.id)
        except Exception as e:
            print(f"Error cancelling batch {batch.id}: {e}")
            continue

def download_batches(output_dir, api_key, description=None):
    # print('description', description)
    os.makedirs(output_dir, exist_ok=True)
    client = OpenAI(api_key=api_key)

    batches = []
    for batch in tqdm(client.batches.list()):
        time.sleep(1)
        batches.append(batch)

    for batch in tqdm(batches):
        desc = batch.metadata['description']
        error_path = os.path.join(output_dir, f'{desc}_error.jsonl')
        output_path = os.path.join(output_dir, f'{desc}_output.jsonl')
        if os.path.exists(error_path) or os.path.exists(output_path):
            # print(f"HAS {batch.id} {batch.status} {batch.metadata['description']}")
            continue
        if description is not None and description not in desc:
            # print(f"NO MATCH {batch.id} {batch.status} {batch.metadata['description']}")
            continue
        if batch.status not in ['completed', 'failed']:
            continue
        # print(f"{batch.id} {batch.status} {batch.metadata['description']}")

        print(batch)
        error_file_id = batch.error_file_id
        output_file_id = batch.output_file_id
        if error_file_id:
            time.sleep(1)
            error_content = client.files.content(error_file_id).content
            with open(error_path, 'wb') as f:
                f.write(error_content)
        if output_file_id:
            time.sleep(1)
            output_content = client.files.content(output_file_id).content
            path = os.path.join(output_dir, f'{desc}_output.jsonl')
            with open(output_path, 'wb') as f:
                f.write(output_content)

def build_csv(input_dir, output_file, _type="video"):
    seen_paths = set()
    counts = collections.Counter()
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([_type, "text"])
        for file in tqdm(sorted(os.listdir(input_dir))):
            if file.endswith("_output.jsonl"):
                with open(os.path.join(input_dir, file), 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        path = data['custom_id']
                        if path in seen_paths:
                            raise ValueError(f"Duplicate path: {path}")
                        seen_paths.add(path)
                        # try:
                        text = data['response']['body']['choices'][0]['message']['content']
                        if 'not enough information' in text.lower():
                            counts['not enough information'] += 1
                            continue
                        elif 'single image' in text.lower():
                            counts['single image'] += 1
                            continue
                        elif 'no movement' in text.lower():
                            counts['no movement'] += 1
                            continue
                        else:
                            counts['good'] += 1
                        writer.writerow([path, text])
                        # except Exception as e:
                        #     print(data.keys())
                        #     print(data)
                        #     print(data['response'].keys())
                        #     print(data['response'])
                        #     print(data['response']['body'].keys())
                        #     print(data['response']['body']['choices'][0]['message']['content'])
                        #     return
                        #     # print(e)
                        #     # raise e
    print(counts)

def csv_migrate_clips_location(source_clips_dir, target_clips_dir, source_csv):
    output_file = source_csv.replace(".csv", "_migrated.csv")
    with open(source_csv, 'r') as f_src, open(output_file, 'w') as f_dst:
        reader = csv.reader(f_src)
        writer = csv.writer(f_dst)
        row = next(reader)
        writer.writerow(row)
        for row in reader:
            path = row[0]
            # Replace source_clips_dir with target_clips_dir in path
            path = os.path.join(target_clips_dir, path.replace(source_clips_dir, ''))
            writer.writerow([path] + row[1:])