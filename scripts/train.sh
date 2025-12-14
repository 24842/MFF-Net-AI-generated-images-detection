#!/bin/bash

# 设置 Python 路径，确保能找到 modules
export PYTHONPATH=$PYTHONPATH:.

DATA_ROOT="/data1/zhanghongji/datasets"

python train.py \
    --sem_train "${DATA_ROOT}/semantic_dinov3_sdv_train.pt" \
    --lrhr_train "${DATA_ROOT}/lrhr_vgg_sdv_train.pt" \
    --label_train "${DATA_ROOT}/label_sdv_tr.pt" \
    --sem_val "${DATA_ROOT}/semantic_dinov3_sdv_val.pt" \
    --lrhr_val "${DATA_ROOT}/lrhr_vgg_sdv_val.pt" \
    --label_val "${DATA_ROOT}/label_sdv_v.pt" \
    --save_dir "./checkpoints/experiment_GenImage" \
    --batch_size 32 \
    --epochs 100 \
    --patience 6 \
    --lr 0.0001