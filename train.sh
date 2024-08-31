export PATH=/home/fan23j/anaconda3/envs/opensora/bin:$PATH

TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=52 colossalai run --nproc_per_node 3 \
scripts/train.py \
configs/opensora-v1-1/train/text2bricks-360p-32f.py \
--data-path  /mnt/mir/fan23j/data/nba-plus-statvu-dataset/__scripts__/filtered_clip_annotations_df_filtered_text.csv \
--ckpt-path pretrained/OpenSora-STDiT-v2-stage3/model.safetensors