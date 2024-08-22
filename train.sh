export PATH=/home/fan23j/anaconda3/envs/opensora/bin:$PATH

TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=52 colossalai run --nproc_per_node 4 \
scripts/train.py \
configs/opensora-v1-1/train/text2bricks-360p-64f.py \
--data-path  data/bball/output_sub_small.csv \
--ckpt-path pretrained/OpenSora-STDiT-v2-stage3/model.safetensors