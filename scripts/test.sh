#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

DATA_ROOT="/data1/zhanghongji/datasets"

python test.py \
    --sem_test "${DATA_ROOT}/semantic_dinov3_Chameleon_test.pt" \
    --lrhr_test "${DATA_ROOT}/lrhr_vgg_Cha_test.pt" \
    --label_test "${DATA_ROOT}/label_Cha_te.pt" \
    --model_path "./checkpoints/experiemnt_GenImage/fusion_model_dinov3_sdv_best.pt" \
    --save_csv "./results/results_Chameleon.csv"