#!/bin/bash
model=create_uni3d

gpu=1
echo "export CUDA_VISIBLE_DEVICES=$gpu"
export CUDA_VISIBLE_DEVICES=${gpu}


clip_model="EVA02-E-14-plus" 
pretrained="./downloads/open_clip_pytorch_model.bin" # or  "laion2b_s9b_b144k"  

if [ "$1" = "giant" ]; then
    pc_model="eva_giant_patch14_560"
    pc_feat_dim=1408
elif [ "$1" = "large" ]; then
    pc_model="eva02_large_patch14_448"
    pc_feat_dim=1024
elif [ "$1" = "base" ]; then
    pc_model="eva02_base_patch14_448"
    pc_feat_dim=768
elif [ "$1" = "small" ]; then
    pc_model="eva02_small_patch14_224"
    pc_feat_dim=384
elif [ "$1" = "tiny" ]; then
    pc_model="eva02_tiny_patch14_224"
    pc_feat_dim=192
else
    echo "Invalid option"
    exit 1
fi

ckpt_path="./downloads/ckpt/model_giant.pt"
# --standalone \
#     --nproc-per-node=1

python -m torch.distributed.run  main.py \
    --model $model \
    --batch-size 32 \
    --npoints 10000 \
    --num-group 512 \
    --group-size 64 \
    --pc-encoder-dim 512 \
    --clip-model $clip_model \
    --pretrained $pretrained \
    --pc-model $pc_model \
    --pc-feat-dim $pc_feat_dim \
    --embed-dim 1024 \
    --validate_dataset_name modelnet40_openshape \
    --validate_dataset_name_lvis objaverse_lvis_openshape \
    --validate_dataset_name_scanobjnn scanobjnn_openshape \
    --evaluate_3d \
    --ckpt_path $ckpt_path \
