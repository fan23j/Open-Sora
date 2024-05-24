# Instructions for batch captioning
This directory contains scripts for batch captioning images using the OpenAI Batch API


## Step 0: Set up the environment variables (in addition to the ones in the [captioning README](/tools/caption/README.md)
```
ROOT_IMAGES=/path/to/images
ROOT_BATCHES=/path/to/batches
JOB_NAME=job_name
```

## Step 1: Create the batch '.jsonl' files for the jobs

### A. Video dataset
```
python -m tools.caption.batch.prepare_batches ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv \
    --prompt "video" \
    --output-dir ${ROOT_BATCHES} \
    --to-images \
    --image-output-dir ${ROOT_IMAGES} \
    --name ${JOB_NAME}
```

### B. Image dataset
```
python -m tools.caption.batch.prepare_batches ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.0.csv \
    --prompt "image" \
    --output-dir ${ROOT_BATCHES} \
    --name ${JOB_NAME}
```

## Step 2: Submit the batch jobs
Make sure to update the path below to the one matching the subdir we just created within the `ROOT_BATCHES` directory
```
python -m tools.caption.batch.submit_batches ${ROOT_BATCHES}/2024-05-21T02_41_32 --key $OPENAI_API_KEY
```

### Step 2b: Optional, check the status of the jobs
```
python -m tools.caption.batch.check_batches --key $OPENAI_API_KEY --desc $JOB_NAME
```


## Step 3: Download the results
Once the jobs are completed, you can download the results (as '.jsonl' files)
```
python tools/caption/batch/download_batches.py ${ROOT_BATCHES}/2024-05-16T19_30_42_output/ \
    --key $OPENAI_API_KEY \
    --desc ${JOB_NAME}
```

## Step 4: Merge the results
Finally, we can merge the results into a single csv file

**If this is an image dataset**: add the flag `--type image`
```
python tools/caption/batch/build_csv.py ${ROOT_BATCHES}/2024-05-16T19_30_42_output/ ${ROOT_META}/meta_clips_caption.csv
```