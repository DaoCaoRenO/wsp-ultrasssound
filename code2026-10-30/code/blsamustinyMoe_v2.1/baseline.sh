# export CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=12346 \
#     omni_train.py \
#     --root_path /root/autodl-tmp/wsp_uusic/blsamus/data \
#     --output_dir /root/autodl-tmp/wsp_uusic/blsamustinyMoe_v2.1/exp_out/2025-8-20/samus_tiny_moe_v2p1_bs16x2_e200_lr1e-5 \
#     --cfg /root/autodl-tmp/wsp_uusic/blsamustinyMoe_v2.1/configs/tiny_samus_config.yaml \
#     --max_epochs 200 \
#     --batch_size 16 \
#     --accumulation-steps 2 \
#     --base_lr 1e-5 \
#     --img_size 224 \
#     --seed 1234 \
#     --deterministic 1 \
#     --use-checkpoint \
#     --amp-opt-level O1 \
#     --prompt \
#     --tag samus_tiny_moe_v2p1_k2_e6_lr1e-5_bs32eff \
#     --opts \
#         MODEL.MOE.NUM_EXPERTS 4 \
#         MODEL.MOE.TOP_K 1 \
#         MODEL.MOE.TEMPERATURE 1.0 \
#         MODEL.MOE.NOISE 0.01 \
#         MODEL.MOE.AUX_WEIGHT 0.05 \
#         MODEL.MOE.LOG_INTERVAL 100



export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12346 \
    omni_test.py \
    --output_dir=/root/autodl-tmp/wsp_uusic/blsamustinyMoe_v2.1/exp_out/2025-8-20/samus_tiny_moe_v2p1_bs16x2_e200_lr1e-5
     \
    --base_lr=0.0003
