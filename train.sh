CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py \
  --model_name /home/zhangping/zrc/fla_models/zerocot_sft \
  --base_model_name /home/zhangping/zrc/fla_models/zerocot_sft \
  --config_args '{"fuse_norm":false,"fuse_cross_entropy":false}' \
  --data_file /home/zhangping/zrc/rwkv-lm/RWKV-v5/data/minipile \
  --ctx_len 4096 \
  --bsz 11 \
  --strategy deepspeed_stage_2_offload \
  --compile_base_model \
  --lr 1e-5 \
  --lr_end 1e-5 \
  --weight_decay 0.001 \
  --warmup_steps 0 \
  --grad_clip_val 4.0 \
  --accelerator gpu \
  --devices 6 \
  --save_every 1 \
  --steps_per_epoch 100 \
  --precision bf16-true \
  --wandb zerocot \
  --beta1 0.9 \
  --beta2 0.95 \
  --rl_coeff 0.001 \
  --rl_gamma 0.99 \
  --epsilon 3.552713678800501e-15 \

  
  # --no_decay_1d
  # --config_args '{"fuse_cross_entropy":true}' \
  # --model_args {"_attn_implementation":"flash_attention_2"} \
  # --data_file /home/zhangping/zrc/RWKV-LM-v6/RWKV-v5/data/rwkv_mypile_v2 \
  # --compile_model \
  # /home/zhangping/zrc/rwkv-lm/RWKV-v5/data/minipile.bin
  # compile_model