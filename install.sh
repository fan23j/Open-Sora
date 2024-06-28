#!/bin/bash
# Description: Setup Open-Sora Environment

ENV_NAME="brick-osora"
PYTHON_VER="python=3.10"
export MAX_JOBS=$(nproc) && echo "MAX_JOBS set to $MAX_JOBS"
CONDA_HOME=$(conda info --base)
CUDA_HOME=$CONDA_HOME

conda_pip() {
    "$CONDA_HOME/envs/$ENV_NAME/bin/python" -m pip "$@"
}

# Function to infer and set CUDA_HOME from the Conda environment
set_cuda_home() {
    # Construct the path to the Conda environment
    ENV_PATH="$CONDA_HOME/envs/$ENV_NAME"

    # Check for common CUDA installation directories within the Conda environment
    if [ -d "$ENV_PATH" ]; then
        # Check for the existence of CUDA bin directory
        if [ -d "$ENV_PATH/lib" ]; then
            export CUDA_HOME="$ENV_PATH"
            export PATH="$CUDA_HOME/bin:$PATH"
            PROFILER_PATH=$(dirname $(find $CUDA_HOME -name "cuda_profiler_api.h"))
            export LD_LIBRARY_PATH="$PROFILER_PATH/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
            export CXXFLAGS="-I$PROFILER_PATH"
            export CPPFLAGS="-I$PROFILER_PATH"
            echo "CUDA_HOME set to $CUDA_HOME"
        else
            echo "CUDA installation not found in Conda environment $ENV_NAME"
        fi
    else
        echo "Conda environment $ENV_NAME not found"
    fi
}

# Create a virtual environment
conda init
conda create -y --name "$ENV_NAME" "$PYTHON_VER"
conda activate "$ENV_NAME"

# Install dependencies with CUDA 12.1.0 via the NVIDIA channel
conda install -n $ENV_NAME -c nvidia cuda-toolkit=12.1.0 \
  cuda-command-line-tools=12.1.0 \
  cuda-compiler=12.1.0 \
  cuda-documentation=12.1.55 \
  cuda-cuxxfilt=12.1.55 \
  cuda-nvcc=12.1.66 \
  cuda-tools=12.1.0 \
  libnvvm-samples=12.1.55 \
  libnvjitlink=12.1.105 \
  cuda-libraries=12.1.0 \
  cuda-libraries-dev=12.1.0 \
  cuda-cccl=12.1.55 \
  cuda-profiler-api=12.1.55 \
  cuda-nvprof=12.1.55

set_cuda_home

# Install PyTorch for CUDA 12.1.0
conda install -n $ENV_NAME pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Optional installations
conda_pip install --force packaging
conda_pip install --force ninja
LD_PRELOAD=$(gcc -print-file-name=libstdc++.so.6) TORCH_CUDA_ARCH_LIST=9.0 PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib" conda_pip install --force --no-deps -U xformers --index-url https://download.pytorch.org/whl/cu121
conda_pip install --force -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' git+https://github.com/NVIDIA/apex.git
FLASH_ATTN_WHEEL=flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/$FLASH_ATTN_WHEEL
conda_pip install $FLASH_ATTN_WHEEL
rm $FLASH_ATTN_WHEEL

# Clone and install Open-Sora
if [ ! -d "Open-Sora" ]; then
    git clone https://github.com/davidh-lambda/Open-Sora
fi
pushd "Open-Sora" || return
    git checkout lambda_bricks
    BUILD_EXT=1 conda_pip install -v .
popd || return

conda_pip install --force protobuf==3.20.3

echo "Open-Sora (lambda_lego branch) environment setup completed. Please check success using check_install.sh"
