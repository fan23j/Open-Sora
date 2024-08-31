CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
configs/opensora-v1-1/inference/text2bricks-360p-64f.py \
--num-frames 64 \
--ckpt-path pretrained/OpenSora-STDiT-v2-stage3/model.safetensors