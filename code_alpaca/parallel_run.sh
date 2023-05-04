python train.py \
  --model_name_or_path "chainyo/alpaca-lora-7b"\
  --data_path ./code_regen.json \
  --output_dir ./result \
  --learning_rate 2e-5 \
  --lr_scheduler_type "cosine" \
  --group_by_length \
  --per_gpu_train_batch_size 4 \
  --per_device_train_batch_size 4 \
  --save_steps 1000 \
  --num_train_epochs 3
  # --train_on_inputs \
  # --lora_target_modules '[q_proj,v_proj]' \
  # --lora_dropout 0.05 \
  # --lora_r 8 \
  # --lora_alpha 16 \
  # --val_set_size 2000 \
  # --cutoff_len 512 \
  # --batch_size 128 \
  # --micro_batch_size 4 \