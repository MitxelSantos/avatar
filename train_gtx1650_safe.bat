@echo off
echo ========================================
echo ENTRENAMIENTO LORA GTX 1650 - FINAL
echo Todas las flags compatibles y probadas
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
  --learning_rate 0.00005 ^
  --unet_lr 0.00005 ^
  --text_encoder_lr 0.00002 ^
  --lr_scheduler "cosine_with_restarts" ^
  --lr_warmup_steps 50 ^
  --lr_scheduler_num_cycles 3 ^
  --optimizer_type "AdamW8bit" ^
  --mixed_precision "fp16" ^
  --gradient_checkpointing ^
  --no_half_vae ^
  --cache_latents ^
  --cache_latents_to_disk ^
  --max_data_loader_n_workers 0 ^
  --max_train_epochs 1 ^
  --seed 42 ^
  --max_grad_norm 0.5 ^
  --caption_extension ".txt" ^
  --save_precision "fp16" ^
  --min_snr_gamma 5 ^
  --noise_offset 0.03 ^
  --output_dir "../clients/Esoterico/models" ^
  --output_name "Esoterico_gtx1650_v1" ^
  --save_model_as "safetensors" ^
  --save_every_n_steps 100 ^
  --logging_dir "../clients/Esoterico/training/logs" ^
  --log_with "tensorboard" ^
  --log_prefix "esoterico"

echo.
echo ========================================
echo ENTRENAMIENTO COMPLETADO
echo ========================================
echo.
echo Modelo: ../clients/Esoterico/models/Esoterico_gtx1650_v1.safetensors
echo Logs: ../clients/Esoterico/training/logs
echo.
echo Para monitorear en tiempo real:
echo tensorboard --logdir ../clients/Esoterico/training/logs
echo.
pause