@echo off
cd kohya_ss

python sdxl_train_network.py ^
  --dataset_config "../clients/Esoterico/training_data/dataset_config.toml" ^
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" ^
  --network_module "networks.lora" ^
  --network_dim 64 ^
  --network_alpha 32 ^
  --resolution "768,768" ^
  --train_batch_size 1 ^
  --max_train_steps 500 ^
  --learning_rate 0.0001 ^
  --lr_scheduler "cosine_with_restarts" ^
  --lr_warmup_steps 50 ^
  --optimizer_type "AdamW8bit" ^
  --mixed_precision "bf16" ^
  --full_bf16 ^
  --gradient_checkpointing ^
  --max_data_loader_n_workers 0 ^
  --seed 42 ^
  --caption_extension ".txt" ^
  --save_precision "bf16" ^
  --output_dir "../clients/Esoterico/models" ^
  --output_name "Esoterico_safe_v1" ^
  --save_model_as "safetensors" ^
  --save_every_n_steps 100 ^
  --logging_dir "../clients/Esoterico/training/logs"

pause