export CUDA_VISIBLE_DEVICES=0
# 使用本地 conda 环境（torch2.8 + CUDA12.8，RTX 4090）
export PATH=/home/linux/anaconda3/envs/torch2.8_cuda12.8/bin:$PATH
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12347 \
    omni_train.py \
    --root_path=./data \
    --cfg=./configs/tiny_samus_config.yaml \
    --output_dir=exp_out/trial_1 \
    --base_lr=1e-5 \



# export CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=12347 \
#     omni_test.py \
#     --output_dir=exp_out/2025-08-09-22-36-25 \
#     --base_lr=0.0003 \
#     --is_saveout \


