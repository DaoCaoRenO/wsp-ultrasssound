export CUDA_VISIBLE_DEVICES=0
# 使用本地 conda 环境（torch2.8 + CUDA12.8，RTX 4090）
export PATH=/home/linux/anaconda3/envs/torch2.8_cuda12.8/bin:$PATH

# ===== 训练（已完成 200 epoch，如需续训取消注释；setsid 后台防 SIGHUP 中断）=====
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=12347 \
#     omni_train.py \
#     --root_path=./data \
#     --cfg=./configs/tiny_samus_config.yaml \
#     --output_dir=exp_out/trial_1 \
#     --batch_size=32 \
#     --base_lr=1e-5 \
#     --resume=exp_out/trial_1/latest.pth

# ===== 测试评估：加载 exp_out/trial_1/best_model.pth，结果追加到 exp_out/result.csv =====
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12347 \
    omni_test.py \
    --root_path=./data \
    --cfg=./configs/tiny_samus_config.yaml \
    --output_dir=exp_out/trial_1 \
    --batch_size=32 \
    --is_saveout \



