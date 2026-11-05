#!/usr/bin/env bash
# =============================================================================
# baseline_nothy.sh — Experiment B: blsamus 训练（无甲状腺：排除 DDTI/private_Thyroid/TN3K）
#   目的：验证纯多器官(乳腺/心脏/胎儿/肾脏)能否零样本泛化到甲状腺。
#   配方：与官方 baseline.sh 一致，仅改 root_path→data_nothy、epochs 200→100。
# 用法：setsid nohup bash baseline_nothy.sh > train_nothy.log 2>&1 &
# =============================================================================
set -u
export CUDA_VISIBLE_DEVICES=0
export PATH=/home/linux/anaconda3/envs/torch2.8_cuda12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")" || exit 1

echo "[bl_nothy] START $(date)  root=data_nothy epochs=100"

python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=22346 \
    omni_train.py \
    --root_path=./data_nothy \
    --cfg=./configs/samus_config.yaml \
    --output_dir=exp_out/samus_b_nothy \
    --batch_size=12 \
    --base_lr=1e-5 \
    --max_epochs=100

echo "[bl_nothy] DONE $(date)"
