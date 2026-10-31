export CUDA_VISIBLE_DEVICES=0
# 使用本地 conda 环境（torch2.8 + CUDA12.8，RTX 4090）
export PATH=/home/linux/anaconda3/envs/torch2.8_cuda12.8/bin:$PATH
# 减少显存碎片，便于塞下更大 batch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 训练：SAMUS vit_b 完整版；数据软链自 fjh_blsamus_tiny/data =====
# batch_size=32：32%6!=0 不触发 lr 自动缩放，base_lr 保持 1e-5
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=22345 \
    omni_train.py \
    --root_path=./data \
    --cfg=./configs/samus_config.yaml \
    --output_dir=exp_out/samus_b_trial \
    --batch_size=32 \
    --base_lr=1e-5 \
    --max_epochs=200

# ===== 测试评估：训练完成后取消注释运行 =====
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=22345 \
#     omni_test.py \
#     --root_path=./data \
#     --cfg=./configs/samus_config.yaml \
#     --output_dir=exp_out/samus_b_trial \
#     --batch_size=32 \
#     --is_saveout
