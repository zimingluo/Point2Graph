#!/bin/bash
model=create_uni3d

gpu=1
echo "export CUDA_VISIBLE_DEVICES=$gpu"
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=12

clip_model="EVA02-E-14-plus" 
ckpt_path="./Uni3D/downloads/ckpt/model_giant.pt"
pretrained="./Uni3D/downloads/open_clip_pytorch_model.bin" # or  "laion2b_s9b_b144k"
size="giant"

if [ $size = "giant" ]; then
    pc_model="eva_giant_patch14_560"
    pc_feat_dim=1408
elif [ $size = "large" ]; then
    pc_model="eva02_large_patch14_448"
    pc_feat_dim=1024
elif [ $size = "base" ]; then
    pc_model="eva02_base_patch14_448"
    pc_feat_dim=768
elif [ $size = "small" ]; then
    pc_model="eva02_small_patch14_224"
    pc_feat_dim=384
elif [ $size = "tiny" ]; then
    pc_model="eva02_tiny_patch14_224"
    pc_feat_dim=192
else
    echo "Invalid option"
    exit 1
fi
eps=0.04
min_points=3
python main.py \
--dataset_name scannet \
--dataset_root_dir ./scannet/ \
--meta_data_dir ./scannet/meta_data/ \
--test_ckpt ./models/scannet_540ep.pth \
--auto_test \
--test_only  \
--conf_thresh 0.01  \
--pc-model $pc_model \
--pc-feat-dim $pc_feat_dim \
--pc-encoder-dim 512 \
--ckpt_path $ckpt_path \
--embed-dim 1024 \
--group-size 64 \
--num-group 512 \
--inference_only \
--npoints 10000 \
--eps $eps \
--min_points $min_points
