python train.py \
  --model_name_or_path "chainyo/alpaca-lora-7b"\
  --data_path ./code_regen.json \
  --output_dir ./result \
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --cutoff_len 512 \
  --val_set_size 2000 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lr_scheduler_type "cosine" \
  --lora_target_modules '[q_proj,v_proj]' \
  --train_on_inputs \
  --group_by_length \
  --save_steps 1000