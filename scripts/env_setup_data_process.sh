#!/bin/bash

# Get the full path of the script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory of the script's directory
OPEN_SORA_DIR="$(dirname "$SCRIPT_PATH")"

python -m venv venv-data-process
source venv-data-process/bin/activate
pip install -e $OPEN_SORA_DIR
pip install -r $OPEN_SORA_DIR/scripts/requirements_data_process.txt
