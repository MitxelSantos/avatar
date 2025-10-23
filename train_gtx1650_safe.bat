@echo off
echo ========================================
echo ENTRENAMIENTO LORA - GTX 1650 SAFE MODE
echo FIX: Bucketing habilitado para imágenes de diferentes tamaños
echo ========================================

cd kohya_ss

python sdxl_train_network.py ^
  --dataset_config "../clients/Esoterico/training_data/dataset_config.toml" ^
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" ^
  --network_module "networks.lora" ^
  --network_dim 32 ^
  --network_alpha 16 ^
  --resolution "768,768" ^
  --enable_bucket ^
  --min_bucket_reso 512 ^
  --max_bucket_reso 1024 ^
  --bucket_reso_steps 64 ^
  --bucket_no_upscale ^
  --train_batch_size 1 ^
  --max_train_steps 500 ^
  --learning_rate 0.0001 ^
  --unet_lr 0.0001 ^
  --text_encoder_lr 0.00005 ^
  --lr_scheduler "cosine_with_restarts" ^
  --lr_warmup_steps 50 ^
  --lr_scheduler_num_cycles 3 ^
  --optimizer_type "AdamW8bit" ^
  --mixed_precision "fp16" ^
  --gradient_checkpointing ^
  --max_data_loader_n_workers 0 ^
  --max_train_epochs 1 ^
  --seed 42 ^
  --caption_extension ".txt" ^
  --save_precision "fp16" ^
  --min_snr_gamma 5 ^
  --output_dir "../clients/Esoterico/models" ^
  --output_name "Esoterico_gtx1650_v1" ^
  --save_model_as "safetensors" ^
  --save_every_n_steps 100 ^
  --logging_dir "../clients/Esoterico/training/logs" ^
  --log_with "tensorboard"

echo.
echo ========================================
echo ENTRENAMIENTO COMPLETADO
echo ========================================
pause