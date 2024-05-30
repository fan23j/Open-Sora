#!/bin/bash

# Get the full path of the script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory of the script's directory
OPEN_SORA_DIR="$(dirname "$SCRIPT_PATH")"

# Get the pretrained weights for the aesthetic model
mkdir -p ${OPEN_SORA_DIR}/pretrained_models
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O ${OPEN_SORA_DIR}/pretrained_models/aesthetic.pth

python -m venv venv-data-process
source venv-data-process/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install -e $OPEN_SORA_DIR
pip install -r $OPEN_SORA_DIR/scripts/requirements_data_process.txt
