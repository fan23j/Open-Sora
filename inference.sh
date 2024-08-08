CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
configs/opensora-v1-1/inference/text2bricks-360p-64f.py \
--num-frames 64 \
--ckpt-path outputs/t2b-bs128-STDiT2-XL-2/epoch1-global_step28501