#!/usr/bin/env python3
"""
lora_trainer.py - Entrenador LoRA profesional con detección automática de GPU
Versión 3.0 - Bug KeyError corregido, arquitectura escalable, RTX 3050 optimizado
"""

import os
import sys
import json
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Imports locales
from config import CONFIG, GPUProfile
from utils import (
    PipelineLogger,
    ProgressTracker,
    save_json_safe,
    load_json_safe,
    estimate_processing_time,
)


class LoRATrainer:
    """Entrenador LoRA profesional con detección automática de hardware"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.logger = PipelineLogger("LoRATrainer", self.config.logs_dir)
        self.kohya_path = None

        # GPU detectada automáticamente
        self.detected_gpu_profile = self.config.detect_gpu_profile()
        if self.detected_gpu_profile:
            self.logger.info(f"GPU detectada: {self.detected_gpu_profile.name}")
        else:
            self.logger.warning("No se pudo detectar GPU compatible")

        # Estado del entrenamiento
        self.training_state = {
            "is_training": False,
            "current_client": None,
            "start_time": None,
            "config_used": None,
        }

    def configure_training(self, client_id: str, clients_dir: Path) -> bool:
        """Configura parámetros de entrenamiento LoRA con detección automática de hardware"""
        self.logger.info(f"Configurando entrenamiento LoRA para cliente: {client_id}")

        client_path = clients_dir / client_id
        dataset_dir = client_path / "dataset_lora"

        # Validaciones iniciales
        if not self._validate_dataset(dataset_dir):
            return False

        # Obtener información del dataset
        dataset_info = self._analyze_dataset(dataset_dir)
        if not dataset_info:
            return False

        # Detectar GPU y mostrar información
        gpu_info = self._get_gpu_info()
        self._display_gpu_info(gpu_info, dataset_info)

        # Mostrar presets disponibles según GPU detectada
        available_presets = self._get_available_presets()
        selected_preset = self._select_training_preset(available_presets, dataset_info)

        if not selected_preset:
            self.logger.info("Configuración cancelada por el usuario")
            return False

        # Generar configuración completa
        full_config = self._generate_training_config(
            client_id, selected_preset, dataset_info, gpu_info
        )

        # Mostrar resumen y confirmar
        if not self._confirm_configuration(full_config, dataset_info, gpu_info):
            return False

        # Guardar configuración
        if self._save_training_config(full_config, client_path):
            # Preguntar si instalar dependencias
            if (
                input("\n¿Verificar e instalar dependencias ahora? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self._setup_training_environment(client_path)
            return True

        return False

    def start_training(self, client_id: str, clients_dir: Path) -> bool:
        """Inicia entrenamiento LoRA con monitoreo completo"""
        self.logger.info(f"Iniciando entrenamiento LoRA para cliente: {client_id}")

        client_path = clients_dir / client_id
        config_file = client_path / "training" / "lora_config.json"

        # Validar configuración
        if not config_file.exists():
            self.logger.error("Configuración de entrenamiento no encontrada")
            print(
                "❌ No hay configuración de entrenamiento. Ejecuta primero la configuración."
            )
            input("Presiona Enter para continuar...")
            return False

        # Cargar configuración
        training_config = load_json_safe(config_file, {}, self.logger)
        if not training_config:
            self.logger.error("Error cargando configuración de entrenamiento")
            return False

        # Validar entorno
        if not self._validate_training_environment():
            return False

        # Mostrar información pre-entrenamiento
        self._display_training_info(training_config, client_path)

        # Confirmar inicio
        if not self._confirm_training_start(training_config):
            return False

        # Ejecutar entrenamiento
        return self._execute_training(training_config, client_path, client_id)

    def show_training_progress(self, client_id: str, clients_dir: Path):
        """Muestra progreso del entrenamiento"""
        client_path = clients_dir / client_id

        print(f"\n📈 PROGRESO DE ENTRENAMIENTO - {client_id}")
        print("=" * 60)

        # Verificar si hay entrenamiento activo
        training_dir = client_path / "training"
        models_dir = client_path / "models"

        if not training_dir.exists():
            print("❌ No se encontró directorio de entrenamiento")
            input("Presiona Enter para continuar...")
            return

        # Buscar logs y modelos
        log_files = list(training_dir.glob("logs/*.log"))
        model_files = (
            list(models_dir.glob("*.safetensors")) if models_dir.exists() else []
        )

        if not log_files and not model_files:
            print("📝 No hay entrenamiento iniciado aún")
            print("\n💡 Para iniciar entrenamiento:")
            print("   1. Configura parámetros de entrenamiento")
            print("   2. Inicia el entrenamiento")
        else:
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                model_time = datetime.fromtimestamp(latest_model.stat().st_mtime)
                model_size = latest_model.stat().st_size / (1024 * 1024)

                print(f"🧠 ÚLTIMO MODELO:")
                print(f"   Archivo: {latest_model.name}")
                print(f"   Tamaño: {model_size:.1f}MB")
                print(f"   Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if log_files:
                print(f"\n📋 LOGS DISPONIBLES:")
                for log_file in sorted(
                    log_files, key=lambda x: x.stat().st_mtime, reverse=True
                )[:3]:
                    log_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    print(
                        f"   {log_file.name} - {log_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            print(f"\n💡 Para monitoreo en tiempo real:")
            print(f"   tensorboard --logdir {training_dir}/logs")
            print(f"   python {client_path}/training/scripts/monitor_training.py")

        input("\nPresiona Enter para continuar...")

    def generate_test_samples(self, client_id: str, clients_dir: Path):
        """Genera muestras de prueba usando el modelo entrenado"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n🎨 GENERACIÓN DE MUESTRAS - {client_id}")
        print("=" * 50)

        if not models_dir.exists():
            print("❌ No se encontró directorio de modelos")
            input("Presiona Enter para continuar...")
            return

        # Buscar modelos disponibles
        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No hay modelos entrenados disponibles")
            print("💡 Completa primero el entrenamiento LoRA")
            input("Presiona Enter para continuar...")
            return

        # Mostrar modelos disponibles
        print("🧠 MODELOS DISPONIBLES:")
        for i, model_file in enumerate(
            sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True), 1
        ):
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            model_size = model_file.stat().st_size / (1024 * 1024)
            print(f"   {i}. {model_file.name}")
            print(
                f"      Tamaño: {model_size:.1f}MB | Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        print(f"\n💡 PARA GENERAR MUESTRAS:")
        print(f"   1. Copia el modelo .safetensors a tu generador de imágenes")
        print(f"   2. Usa el trigger word: '{client_id}'")
        print(f"   3. Weight recomendado: 0.7-1.0")
        print(f"   4. Prompts ejemplo:")
        print(
            f"      - 'portrait of {client_id}, detailed face, professional lighting'"
        )
        print(f"      - '{client_id} smiling, high quality photography'")
        print(f"      - 'close-up of {client_id}, artistic portrait'")

        input("\nPresiona Enter para continuar...")

    def manage_trained_models(self, client_id: str, clients_dir: Path):
        """Gestiona modelos entrenados"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n📦 GESTIÓN DE MODELOS - {client_id}")
        print("=" * 45)

        if not models_dir.exists():
            print("❌ No se encontró directorio de modelos")
            input("Presiona Enter para continuar...")
            return

        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No hay modelos disponibles")
            input("Presiona Enter para continuar...")
            return

        # Mostrar estadísticas de modelos
        total_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        print(f"📊 ESTADÍSTICAS:")
        print(f"   Total modelos: {len(model_files)}")
        print(f"   Espacio usado: {total_size:.1f}MB")

        print(f"\n🧠 MODELOS DISPONIBLES:")
        for i, model_file in enumerate(
            sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True), 1
        ):
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            model_size = model_file.stat().st_size / (1024 * 1024)

            # Extraer información del nombre del archivo
            if "-" in model_file.stem:
                parts = model_file.stem.split("-")
                if parts[-1].isdigit():
                    step_count = parts[-1]
                else:
                    step_count = "final"
            else:
                step_count = "unknown"

            print(f"   {i}. {model_file.name}")
            print(f"      Steps: {step_count} | Tamaño: {model_size:.1f}MB")
            print(f"      Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n💡 OPCIONES:")
        print(f"   1. Los modelos están listos para usar")
        print(f"   2. Copia a tu generador de imágenes favorito")
        print(f"   3. Usa trigger word: '{client_id}'")

        # Opción para limpiar modelos antiguos
        if len(model_files) > 3:
            cleanup = input(f"\n¿Limpiar modelos antiguos? (s/n): ").lower().strip()
            if cleanup.startswith("s"):
                self._cleanup_old_models(model_files, keep_count=3)

        input("\nPresiona Enter para continuar...")

    # Métodos privados de implementación

    def _validate_dataset(self, dataset_dir: Path) -> bool:
        """Valida que el dataset esté listo"""
        if not dataset_dir.exists():
            self.logger.error("Dataset no encontrado")
            print("❌ No hay dataset procesado. Prepara primero el dataset LoRA.")
            input("Presiona Enter para continuar...")
            return False

        # Contar imágenes
        dataset_images = list(dataset_dir.glob("*.png"))
        if len(dataset_images) < 30:
            self.logger.warning(f"Dataset pequeño: {len(dataset_images)} imágenes")
            print(
                f"⚠️ Dataset pequeño ({len(dataset_images)} imágenes). Recomendado mínimo: 50"
            )
            proceed = input("¿Continuar de todos modos? (s/n): ").lower().strip()
            if not proceed.startswith("s"):
                return False

        return True

    def _analyze_dataset(self, dataset_dir: Path) -> Optional[Dict[str, Any]]:
        """Analiza el dataset y retorna información detallada"""
        try:
            # Contar imágenes por tipo
            all_images = list(dataset_dir.glob("*.png"))
            mj_images = [img for img in all_images if "_mj_" in img.name]
            real_images = [img for img in all_images if "_real_" in img.name]

            # Cargar configuración del dataset si existe
            config_file = dataset_dir / "dataset_config.json"
            dataset_config = load_json_safe(config_file, {}, self.logger)

            info = {
                "total_images": len(all_images),
                "mj_images": len(mj_images),
                "real_images": len(real_images),
                "avatar_type": dataset_config.get("avatar_type", "unknown"),
                "distribution": dataset_config.get("balance_ratio", "unknown"),
                "has_captions": len(list(dataset_dir.glob("*.txt"))) == len(all_images),
                "avg_file_size_mb": (
                    sum(img.stat().st_size for img in all_images)
                    / len(all_images)
                    / (1024 * 1024)
                    if all_images
                    else 0
                ),
            }

            return info

        except Exception as e:
            self.logger.error(f"Error analizando dataset: {e}")
            return None

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información detallada de la GPU"""
        if self.detected_gpu_profile:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    return {
                        "profile": self.detected_gpu_profile,
                        "name": torch.cuda.get_device_name(0),
                        "vram_gb": gpu_props.total_memory / (1024**3),
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "multiprocessors": gpu_props.multi_processor_count,
                        "available": True,
                    }
            except ImportError:
                pass

        return {
            "profile": None,
            "name": "Unknown GPU",
            "vram_gb": 0,
            "available": False,
            "error": "No GPU detectada o PyTorch no disponible",
        }

    def _display_gpu_info(self, gpu_info: Dict, dataset_info: Dict):
        """Muestra información de GPU y dataset"""
        print(f"\n⚙️ CONFIGURACIÓN ENTRENAMIENTO LoRA")
        print("=" * 50)

        if gpu_info["available"]:
            print(f"🎮 GPU Detectada: {gpu_info['name']} ({gpu_info['vram_gb']:.1f}GB)")
            if gpu_info["profile"]:
                print(f"📋 Perfil: {gpu_info['profile'].name}")
                print(f"🚀 Optimizaciones específicas activadas")
        else:
            print(f"❌ GPU: {gpu_info.get('error', 'No disponible')}")

        print(f"\n📊 INFORMACIÓN DEL DATASET:")
        print(f"   Total imágenes: {dataset_info['total_images']}")
        print(f"   🎨 MJ: {dataset_info['mj_images']}")
        print(f"   📷 Real: {dataset_info['real_images']}")
        print(f"   Tipo avatar: {dataset_info['avatar_type']}")
        print(f"   Distribución: {dataset_info['distribution']}")
        print(f"   Captions: {'✅' if dataset_info['has_captions'] else '❌'}")

    def _get_available_presets(self) -> List[Dict[str, Any]]:
        """Obtiene presets disponibles según la GPU detectada"""
        presets = []

        # Agregar presets de configuración global
        for preset_key, preset_config in self.config.training_presets.items():
            preset_info = preset_config.copy()
            preset_info["key"] = preset_key

            # Marcar si es recomendado para la GPU actual
            if self.detected_gpu_profile:
                # Presets más adecuados según GPU
                if (
                    preset_key == "balanced"
                    and "3050" in self.detected_gpu_profile.name.lower()
                ):
                    preset_info["recommended"] = True
                elif (
                    preset_key == "quality"
                    and self.detected_gpu_profile.vram_gb_min >= 8
                ):
                    preset_info["recommended"] = True
                elif (
                    preset_key == "quick" and self.detected_gpu_profile.vram_gb_min < 6
                ):
                    preset_info["recommended"] = True
                else:
                    preset_info["recommended"] = False
            else:
                preset_info["recommended"] = False

            presets.append(preset_info)

        return presets

    def _select_training_preset(
        self, presets: List[Dict], dataset_info: Dict
    ) -> Optional[Dict]:
        """Permite al usuario seleccionar un preset de entrenamiento"""
        print(f"\n🎯 PRESETS DE ENTRENAMIENTO DISPONIBLES:")

        for i, preset in enumerate(presets, 1):
            recommended_mark = " 👈 RECOMENDADO" if preset.get("recommended") else ""
            print(f"\n{i}. {preset['name']}{recommended_mark}")
            print(f"   Descripción: {preset['description']}")
            print(f"   Steps: {preset['max_train_steps']}")
            print(f"   Learning Rate: {preset['learning_rate']}")

            # Estimar tiempo
            if self.detected_gpu_profile:
                estimated_time = estimate_processing_time(
                    preset["max_train_steps"],
                    None,  # No necesitamos profile string aquí
                    "lora_training_per_step",
                )
                # Ajustar por GPU
                gpu_multiplier = (
                    3600 / self.detected_gpu_profile.steps_per_hour_estimate
                )
                total_hours = preset["max_train_steps"] * gpu_multiplier
                print(f"   Tiempo estimado: {total_hours:.1f} horas")
            else:
                print(f"   Tiempo estimado: No disponible")

        print(f"\n{len(presets) + 1}. ⚙️ Configuración personalizada (avanzado)")
        print(f"{len(presets) + 2}. 🔙 Cancelar")

        # Selección
        while True:
            try:
                choice = int(input(f"\nSelecciona preset (1-{len(presets) + 2}): "))

                if choice == len(presets) + 2:  # Cancelar
                    return None
                elif choice == len(presets) + 1:  # Personalizada
                    return self._create_custom_preset(dataset_info)
                elif 1 <= choice <= len(presets):
                    return presets[choice - 1]
                else:
                    print("❌ Opción inválida")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                return None

    def _create_custom_preset(self, dataset_info: Dict) -> Dict[str, Any]:
        """Crea preset personalizado con validaciones"""
        print(f"\n⚙️ CONFIGURACIÓN PERSONALIZADA")
        print("-" * 40)

        if self.detected_gpu_profile:
            print(f"GPU: {self.detected_gpu_profile.name}")
            max_recommended_steps = min(5000, dataset_info["total_images"] * 50)
        else:
            print("⚠️ Sin GPU detectada - limitando opciones")
            max_recommended_steps = min(2000, dataset_info["total_images"] * 20)

        preset = {"name": "Configuración Personalizada", "key": "custom"}

        # Steps de entrenamiento
        while True:
            try:
                default_steps = min(
                    max_recommended_steps, dataset_info["total_images"] * 30
                )
                steps_input = input(
                    f"Steps de entrenamiento (recomendado {default_steps}, máx {max_recommended_steps}): "
                ).strip()

                steps = int(steps_input) if steps_input else default_steps
                if 500 <= steps <= max_recommended_steps:
                    preset["max_train_steps"] = steps
                    break
                else:
                    print(f"❌ Rango válido: 500-{max_recommended_steps} steps")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Learning rate
        while True:
            try:
                lr_input = input("Learning rate (recomendado 0.0001): ").strip()
                lr = float(lr_input) if lr_input else 0.0001
                if 0.00005 <= lr <= 0.0005:
                    preset["learning_rate"] = lr
                    break
                else:
                    print("❌ Rango válido: 0.00005-0.0005")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Dataset repeats
        default_repeats = max(50, min(300, 1000 // dataset_info["total_images"]))
        while True:
            try:
                repeats_input = input(
                    f"Dataset repeats (recomendado {default_repeats}): "
                ).strip()
                repeats = int(repeats_input) if repeats_input else default_repeats
                if 30 <= repeats <= 500:
                    preset["dataset_repeats_multiplier"] = repeats
                    break
                else:
                    print("❌ Rango válido: 30-500")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Save frequency
        preset["save_every_n_steps"] = max(200, preset["max_train_steps"] // 6)
        preset["description"] = (
            f"Configuración personalizada - {preset['max_train_steps']} steps"
        )

        return preset

    def _generate_training_config(
        self, client_id: str, preset: Dict, dataset_info: Dict, gpu_info: Dict
    ) -> Dict[str, Any]:
        """Genera configuración completa de entrenamiento"""

        # Usar perfil de GPU detectado o fallback
        gpu_profile = gpu_info.get("profile") or self.config.gpu_profiles["low_end"]

        # Calcular dataset repeats basado en tamaño
        dataset_repeats = max(
            50,
            min(
                preset.get("dataset_repeats_multiplier", 150),
                preset.get("dataset_repeats_multiplier", 150)
                * 100
                // dataset_info["total_images"],
            ),
        )

        config = {
            # Información básica
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "preset_name": preset["name"],
            "preset_key": preset["key"],
            "dataset_size": dataset_info["total_images"],
            "gpu_optimization": f"{gpu_profile.name.replace(' ', '_')}_{gpu_info.get('vram_gb', 0):.0f}GB",
            "detected_gpu": gpu_info.get("name", "Unknown"),
            "avatar_type": dataset_info.get("avatar_type", "unknown"),
            # Modelo base
            "model_config": {
                "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae": "madebyollin/sdxl-vae-fp16-fix",
                "v2": False,
                "v_parameterization": False,
                "clip_skip": 2,
            },
            # Configuración LoRA adaptativa
            "network_config": {
                "network_module": "networks.lora",
                "network_dim": gpu_profile.network_dim,
                "network_alpha": gpu_profile.network_alpha,
                "network_args": {
                    "conv_lora": gpu_profile.conv_lora,
                    "algo": "lora",
                },
            },
            # Optimizaciones de memoria específicas por GPU
            "memory_optimizations": gpu_profile.memory_optimizations.copy(),
            # Configuración de entrenamiento
            "training_config": {
                "max_train_steps": preset["max_train_steps"],
                "learning_rate": preset["learning_rate"],
                "train_batch_size": gpu_profile.batch_size,
                "lr_scheduler": "cosine_with_restarts",
                "lr_warmup_steps": int(preset["max_train_steps"] * 0.1),
                "optimizer_type": gpu_profile.optimizer,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": gpu_profile.gradient_accumulation_steps,
            },
            # Configuración del dataset
            "dataset_config": {
                "resolution": gpu_profile.resolution,
                "bucket_resolution_steps": 64,
                "bucket_no_upscale": True,
                "min_bucket_reso": 512,
                "max_bucket_reso": gpu_profile.resolution,
                "dataset_repeats": dataset_repeats,
                "shuffle_caption": True,
                "caption_extension": ".txt",
                "keep_tokens": 1,
            },
            # Técnicas avanzadas (según GPU)
            "advanced_config": {
                "noise_offset": 0.1 if gpu_profile.conv_lora else 0.05,
                "adaptive_noise_scale": 0.05 if gpu_profile.conv_lora else None,
                "multires_noise_iterations": 4 if gpu_profile.conv_lora else 0,
                "multires_noise_discount": 0.3 if gpu_profile.conv_lora else 0,
                "ip_noise_gamma": None,
                "debiased_estimation_loss": gpu_profile.conv_lora,
            },
            # Configuración de guardado
            "save_config": {
                "save_every_n_steps": preset.get("save_every_n_steps", 500),
                "save_model_as": "safetensors",
                "save_precision": "fp16",
                "output_name": f"{client_id}_avatar_lora_{preset['key']}",
                "max_checkpoints": 5 if gpu_profile.vram_gb_min >= 8 else 3,
            },
            # Configuración de muestras
            "sample_config": {
                "sample_every_n_steps": max(300, preset["max_train_steps"] // 8),
                "sample_sampler": "euler_a",
                "sample_cfg_scale": 7.0,
                "sample_steps": 20,
                "sample_prompts": [
                    f"portrait of {client_id}, detailed face, professional lighting",
                    f"{client_id} smiling, high quality photography",
                    f"close-up of {client_id}, artistic portrait, dramatic lighting",
                ],
            },
            # Logging
            "logging_config": {
                "log_with": "tensorboard",
                "log_tracker_name": f"{client_id}_lora_{preset['key']}",
                "logging_dir": f"./logs/{client_id}",
                "log_level": "INFO",
            },
        }

        return config

    def _confirm_configuration(
        self, config: Dict, dataset_info: Dict, gpu_info: Dict
    ) -> bool:
        """Muestra resumen de configuración y pide confirmación"""
        print(f"\n📋 RESUMEN DE CONFIGURACIÓN")
        print("=" * 50)
        print(f"Cliente: {config['client_id']}")
        print(f"Preset: {config['preset_name']}")
        print(f"Dataset: {dataset_info['total_images']} imágenes")
        print(f"GPU: {gpu_info.get('name', 'Unknown')}")

        print(f"\n🎯 PARÁMETROS DE ENTRENAMIENTO:")
        training_config = config["training_config"]
        print(f"   Steps totales: {training_config['max_train_steps']:,}")
        print(f"   Learning rate: {training_config['learning_rate']}")
        print(f"   Batch size: {training_config['train_batch_size']}")
        print(f"   Network dim: {config['network_config']['network_dim']}")
        print(f"   Resolución: {config['dataset_config']['resolution']}px")
        print(f"   Dataset repeats: {config['dataset_config']['dataset_repeats']}")

        print(f"\n🚨 OPTIMIZACIONES ACTIVADAS:")
        memory_opts = config["memory_optimizations"]
        key_optimizations = [
            "mixed_precision",
            "gradient_checkpointing",
            "cache_latents",
            "xformers_memory_efficient_attention",
        ]
        for opt in key_optimizations:
            if memory_opts.get(opt):
                readable_name = opt.replace("_", " ").title()
                print(f"   ✅ {readable_name}")

        # Estimación de tiempo
        if gpu_info.get("profile"):
            total_hours = (
                training_config["max_train_steps"]
                / gpu_info["profile"].steps_per_hour_estimate
            )
            print(f"\n⏱️ TIEMPO ESTIMADO:")
            print(f"   Duración: {total_hours:.1f} horas")
            print(
                f"   Velocidad: ~{gpu_info['profile'].steps_per_hour_estimate} steps/hora"
            )

        print(f"\n💾 GUARDADO:")
        print(f"   Cada {config['save_config']['save_every_n_steps']} steps")
        print(f"   Máximo {config['save_config']['max_checkpoints']} checkpoints")

        return input("\n¿Guardar esta configuración? (s/n): ").lower().startswith("s")

    def _save_training_config(self, config: Dict, client_path: Path) -> bool:
        """Guarda configuración de entrenamiento"""
        try:
            config_file = client_path / "training" / "lora_config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            if save_json_safe(config, config_file, self.logger):
                self.logger.info(f"Configuración guardada: {config_file}")
                print(f"✅ Configuración guardada en: {config_file}")
                return True
            else:
                print("❌ Error guardando configuración")
                return False

        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")
            print(f"❌ Error guardando configuración: {e}")
            return False

    def _setup_training_environment(self, client_path: Path) -> bool:
        """Configura entorno de entrenamiento"""
        print(f"\n🔧 CONFIGURANDO ENTORNO DE ENTRENAMIENTO")
        print("=" * 50)

        # Verificar Python y dependencias básicas
        if not self._verify_python_environment():
            return False

        # Verificar CUDA
        if not self._verify_cuda():
            return False

        # Configurar Kohya_ss
        if not self._setup_kohya_ss():
            return False

        # Crear scripts de entrenamiento
        self._create_training_scripts(client_path)

        print(f"\n✅ ENTORNO CONFIGURADO CORRECTAMENTE")
        print(f"📋 Scripts de entrenamiento creados")
        print(f"🚀 Sistema listo para entrenar")

        input("Presiona Enter para continuar...")
        return True

    def _verify_python_environment(self) -> bool:
        """Verifica entorno Python"""
        try:
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 8:
                print(
                    f"❌ Python 3.8+ requerido. Actual: {python_version.major}.{python_version.minor}"
                )
                return False

            print(
                f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )

            # Verificar dependencias críticas
            critical_deps = [
                "torch",
                "diffusers",
                "transformers",
                "accelerate",
                "safetensors",
            ]
            missing = []

            for dep in critical_deps:
                try:
                    __import__(dep)
                    print(f"✅ {dep}")
                except ImportError:
                    missing.append(dep)
                    print(f"❌ {dep}")

            if missing:
                print(f"\n🔧 Dependencias faltantes: {', '.join(missing)}")
                print(f"💡 Instalar con: pip install {' '.join(missing)}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error verificando Python: {e}")
            return False

    def _verify_cuda(self) -> bool:
        """Verifica instalación de CUDA"""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                print(f"✅ GPU: {gpu_name}")
                print(f"✅ VRAM: {vram_gb:.1f}GB")
                print(f"✅ CUDA {torch.version.cuda}")

                if vram_gb < 4:
                    print(f"⚠️ VRAM muy baja - entrenamiento será extremadamente lento")

                return True
            else:
                print(f"❌ CUDA no disponible. Se requiere GPU para entrenamiento.")
                return False

        except ImportError:
            print(f"❌ PyTorch no instalado.")
            return False

    def _setup_kohya_ss(self) -> bool:
        """Configura Kohya_ss para entrenamiento"""
        print(f"\n🎯 CONFIGURANDO KOHYA_SS...")

        # Crear directorio para Kohya_ss
        kohya_dir = Path("./kohya_ss")

        if not kohya_dir.exists():
            print(f"📥 Clonando Kohya_ss...")
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "https://github.com/kohya-ss/sd-scripts.git",
                        str(kohya_dir),
                    ],
                    check=True,
                )
                print(f"✅ Kohya_ss clonado exitosamente")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error clonando Kohya_ss: {e}")
                return False
            except FileNotFoundError:
                print(f"❌ Git no encontrado. Instala Git primero.")
                return False

        # Verificar archivos críticos
        train_script = kohya_dir / "train_network.py"
        if not train_script.exists():
            print(f"❌ Script de entrenamiento no encontrado en Kohya_ss")
            return False

        print(f"✅ Kohya_ss configurado correctamente")
        self.kohya_path = kohya_dir
        return True

    def _create_training_scripts(self, client_path: Path):
        """Crea scripts de entrenamiento optimizados"""
        scripts_dir = client_path / "training" / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Script principal de entrenamiento
        train_script = scripts_dir / "train_lora.py"
        self._create_main_training_script(train_script, client_path)

        # Script de monitoreo
        monitor_script = scripts_dir / "monitor_training.py"
        self._create_monitoring_script(monitor_script)

        # Batch file para Windows
        batch_file = scripts_dir / "train_lora.bat"
        self._create_batch_script(batch_file, client_path)

        # Script de validación
        validate_script = scripts_dir / "validate_setup.py"
        self._create_validation_script(validate_script)

        self.logger.info(f"Scripts de entrenamiento creados en: {scripts_dir}")

    def _create_main_training_script(self, script_path: Path, client_path: Path):
        """Crea script principal de entrenamiento"""
        script_content = f'''#!/usr/bin/env python3
"""
train_lora.py - Script de entrenamiento LoRA
Generado automáticamente por Avatar Pipeline v3.0
Cliente: {client_path.name}
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_training_command(config, dataset_path, output_path):
    memory_opts = config["memory_optimizations"]
    training_opts = config["training_config"]
    network_opts = config["network_config"]
    dataset_opts = config["dataset_config"]
    
    cmd = [
        sys.executable,
        "train_network.py",
        
        # Modelo y VAE
        f"--pretrained_model_name_or_path={{config['model_config']['pretrained_model_name_or_path']}}",
        f"--vae={{config['model_config']['vae']}}",
        
        # Dataset
        f"--train_data_dir={{dataset_path}}",
        f"--resolution={{dataset_opts['resolution']}}",
        f"--train_batch_size={{training_opts['train_batch_size']}}",
        f"--dataset_repeats={{dataset_opts['dataset_repeats']}}",
        
        # Red LoRA
        f"--network_module={{network_opts['network_module']}}",
        f"--network_dim={{network_opts['network_dim']}}",
        f"--network_alpha={{network_opts['network_alpha']}}",
        
        # Entrenamiento
        f"--max_train_steps={{training_opts['max_train_steps']}}",
        f"--learning_rate={{training_opts['learning_rate']}}",
        f"--lr_scheduler={{training_opts['lr_scheduler']}}",
        f"--lr_warmup_steps={{training_opts['lr_warmup_steps']}}",
        f"--optimizer_type={{training_opts['optimizer_type']}}",
        f"--gradient_accumulation_steps={{training_opts['gradient_accumulation_steps']}}",
        
        # Optimizaciones de memoria
        f"--mixed_precision={{memory_opts['mixed_precision']}}",
        
        # Guardado
        f"--output_dir={{output_path}}",
        f"--output_name={{config['save_config']['output_name']}}",
        f"--save_model_as={{config['save_config']['save_model_as']}}",
        f"--save_every_n_steps={{config['save_config']['save_every_n_steps']}}",
        
        # Logging
        f"--logging_dir={{config['logging_config']['logging_dir']}}",
        f"--log_with={{config['logging_config']['log_with']}}",
        
        # Flags de optimización
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--xformers",
        "--bucket_no_upscale",
        "--shuffle_caption",
        "--caption_extension=.txt",
        "--keep_tokens=1",
    ]
    
    # Flags condicionales
    if memory_opts.get("lowvram"):
        cmd.append("--lowvram")
    if memory_opts.get("medvram"):
        cmd.append("--medvram")
    
    # Técnicas avanzadas
    advanced = config.get("advanced_config", {{}})
    if advanced.get("noise_offset"):
        cmd.extend(["--noise_offset", str(advanced["noise_offset"])])
    if advanced.get("adaptive_noise_scale"):
        cmd.extend(["--adaptive_noise_scale", str(advanced["adaptive_noise_scale"])])
    if advanced.get("multires_noise_iterations"):
        cmd.extend(["--multires_noise_iterations", str(advanced["multires_noise_iterations"])])
    
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Archivo de configuración")
    parser.add_argument("--dataset", required=True, help="Directorio del dataset")
    parser.add_argument("--output", required=True, help="Directorio de salida")
    
    args = parser.parse_args()
    
    print("🚀 AVATAR PIPELINE - ENTRENAMIENTO LoRA")
    print("=" * 60)
    
    # Cargar configuración
    config = load_config(args.config)
    print(f"✅ Configuración: {{config.get('preset_name', 'Unknown')}}")
    
    # Cambiar al directorio de Kohya_ss
    kohya_path = Path("./kohya_ss")
    if not kohya_path.exists():
        print("❌ Kohya_ss no encontrado")
        sys.exit(1)
    
    original_cwd = os.getcwd()
    os.chdir(kohya_path)
    
    try:
        # Construir y ejecutar comando
        cmd = build_training_command(config, args.dataset, args.output)
        print(f"\\n🚀 Iniciando entrenamiento...")
        print(f"Hora inicio: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"\\n🎉 ¡Entrenamiento completado!")
        else:
            print(f"\\n❌ Entrenamiento falló")
            sys.exit(result.returncode)
            
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
'''

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

    def _create_monitoring_script(self, script_path: Path):
        """Crea script de monitoreo de entrenamiento"""
        script_content = '''#!/usr/bin/env python3
"""
monitor_training.py - Monitor de progreso de entrenamiento LoRA
"""

import os
import time
import re
from pathlib import Path
from datetime import datetime

def monitor_training(output_dir, log_dir=None):
    print("🔍 Monitoreando entrenamiento...")
    print("Presiona Ctrl+C para salir")
    
    output_path = Path(output_dir)
    last_step = 0
    start_time = time.time()
    
    try:
        while True:
            # Buscar checkpoints más recientes
            checkpoints = list(output_path.glob("*.safetensors"))
            
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                
                # Extraer step del nombre
                match = re.search(r'-(\d+)\.safetensors$', latest_checkpoint.name)
                if match:
                    current_step = int(match.group(1))
                    
                    if current_step > last_step:
                        elapsed = time.time() - start_time
                        size_mb = latest_checkpoint.stat().st_size / (1024 * 1024)
                        
                        print(f"Step {current_step:,} | "
                              f"Tiempo: {elapsed/3600:.1f}h | "
                              f"Modelo: {size_mb:.1f}MB | "
                              f"{datetime.now().strftime('%H:%M:%S')}")
                        
                        last_step = current_step
            
            time.sleep(30)  # Verificar cada 30 segundos
            
    except KeyboardInterrupt:
        print("\\n🛑 Monitoreo detenido")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python monitor_training.py <output_dir>")
        sys.exit(1)
    
    monitor_training(sys.argv[1])
'''

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

    def _create_batch_script(self, script_path: Path, client_path: Path):
        """Crea script batch para Windows"""
        batch_content = f"""@echo off
echo 🚀 Avatar Pipeline - Entrenamiento LoRA
echo Cliente: {client_path.name}
echo ================================================================

if not exist "kohya_ss" (
    echo ❌ Kohya_ss no encontrado
    pause
    exit /b 1
)

if exist ".venv\\Scripts\\activate.bat" (
    call .venv\\Scripts\\activate.bat
)

python "{client_path}\\training\\scripts\\train_lora.py" ^
    --config "{client_path}\\training\\lora_config.json" ^
    --dataset "{client_path}\\dataset_lora" ^
    --output "{client_path}\\models"

if %ERRORLEVEL% EQU 0 (
    echo ✅ ¡Entrenamiento completado!
    echo 📦 Modelos en: {client_path}\\models
) else (
    echo ❌ Entrenamiento falló
)

pause
"""

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(batch_content)

    def _create_validation_script(self, script_path: Path):
        """Crea script de validación del entorno"""
        script_content = '''#!/usr/bin/env python3
"""
validate_setup.py - Validación del entorno de entrenamiento
"""

import sys
from pathlib import Path

def main():
    print("🔍 VALIDACIÓN DEL ENTORNO")
    print("=" * 40)
    
    issues = []
    
    # Python
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ requerido")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # PyTorch y CUDA
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name} ({vram:.1f}GB)")
        else:
            issues.append("CUDA no disponible")
    except ImportError:
        issues.append("PyTorch no instalado")
    
    # Dependencias críticas
    deps = ['diffusers', 'transformers', 'accelerate', 'safetensors']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            issues.append(f"{dep} no instalado")
    
    # Kohya_ss
    if Path("./kohya_ss").exists():
        print("✅ kohya_ss")
    else:
        issues.append("kohya_ss no encontrado")
    
    # Resultado
    if not issues:
        print("\\n🎉 ¡ENTORNO LISTO!")
    else:
        print(f"\\n❌ PROBLEMAS:")
        for issue in issues:
            print(f"   - {issue}")

if __name__ == "__main__":
    main()
'''

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

    def _validate_training_environment(self) -> bool:
        """Valida que el entorno esté listo para entrenamiento"""
        try:
            # Verificar PyTorch
            import torch

            if not torch.cuda.is_available():
                print("❌ CUDA no disponible")
                return False

            # Verificar Kohya_ss
            if not self.kohya_path or not self.kohya_path.exists():
                print("❌ Kohya_ss no configurado")
                return False

            return True

        except ImportError:
            print("❌ PyTorch no disponible")
            return False

    def _display_training_info(self, config: Dict, client_path: Path):
        """Muestra información pre-entrenamiento"""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO LORA")
        print("=" * 50)
        print(f"Cliente: {config['client_id']}")
        print(f"Preset: {config['preset_name']}")
        print(f"GPU: {config.get('detected_gpu', 'Unknown')}")

        training_config = config["training_config"]
        print(f"\nPARÁMETROS:")
        print(f"  Steps: {training_config['max_train_steps']:,}")
        print(f"  Learning Rate: {training_config['learning_rate']}")
        print(f"  Batch Size: {training_config['train_batch_size']}")
        print(f"  Network Dim: {config['network_config']['network_dim']}")

    def _confirm_training_start(self, config: Dict) -> bool:
        """Confirma el inicio del entrenamiento"""
        training_config = config["training_config"]

        # Estimar tiempo basado en GPU detectada
        if self.detected_gpu_profile:
            estimated_hours = (
                training_config["max_train_steps"]
                / self.detected_gpu_profile.steps_per_hour_estimate
            )
        else:
            estimated_hours = training_config["max_train_steps"] / 200  # Conservador

        print(f"\n🚨 CONFIRMACIÓN DE INICIO")
        print(f"Duración estimada: {estimated_hours:.1f} horas")
        print(f"GPU requerida: CUDA compatible")
        print(f"🔥 El sistema estará ocupado durante el entrenamiento")

        confirm = (
            input("\n¿Iniciar entrenamiento? Escribe 'SI' para confirmar: ")
            .strip()
            .upper()
        )
        return confirm == "SI"

    def _execute_training(
        self, config: Dict, client_path: Path, client_id: str
    ) -> bool:
        """Ejecuta el entrenamiento LoRA"""
        try:
            # Actualizar estado
            self.training_state.update(
                {
                    "is_training": True,
                    "current_client": client_id,
                    "start_time": datetime.now(),
                    "config_used": config,
                }
            )

            print(f"\n🚀 EJECUTANDO ENTRENAMIENTO...")
            print(
                f"Hora inicio: {self.training_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Crear directorios necesarios
            models_dir = client_path / "models"
            logs_dir = client_path / "training" / "logs"
            models_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Ejecutar script de entrenamiento
            script_path = client_path / "training" / "scripts" / "train_lora.py"

            if not script_path.exists():
                print("❌ Script de entrenamiento no encontrado")
                return False

            # Construir comando
            cmd = [
                sys.executable,
                str(script_path),
                "--config",
                str(client_path / "training" / "lora_config.json"),
                "--dataset",
                str(client_path / "dataset_lora"),
                "--output",
                str(models_dir),
            ]

            # Ejecutar
            result = subprocess.run(cmd)

            # Actualizar estado
            self.training_state["is_training"] = False

            if result.returncode == 0:
                print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
                end_time = datetime.now()
                duration = end_time - self.training_state["start_time"]
                print(f"Duración total: {duration}")
                print(f"Modelos guardados en: {models_dir}")

                # Registrar en historial del cliente
                self._update_training_history(client_path, config, duration, True)

                return True
            else:
                print(f"\n❌ ENTRENAMIENTO FALLÓ")
                print(f"Código de error: {result.returncode}")

                # Registrar fallo en historial
                self._update_training_history(client_path, config, None, False)

                return False

        except Exception as e:
            self.logger.error(f"Error ejecutando entrenamiento: {e}")
            print(f"❌ Error inesperado: {e}")
            self.training_state["is_training"] = False
            return False

    def _update_training_history(
        self, client_path: Path, config: Dict, duration, success: bool
    ):
        """Actualiza historial de entrenamientos del cliente"""
        try:
            config_file = client_path / "metadata" / "client_config.json"
            client_config = load_json_safe(config_file, {})

            if "training_history" not in client_config:
                client_config["training_history"] = []

            history_entry = {
                "date": datetime.now().isoformat(),
                "preset": config.get("preset_name", "Unknown"),
                "steps": config.get("training_config", {}).get("max_train_steps", 0),
                "success": success,
                "duration_seconds": duration.total_seconds() if duration else None,
                "gpu_used": config.get("detected_gpu", "Unknown"),
            }

            client_config["training_history"].append(history_entry)
            save_json_safe(client_config, config_file, self.logger)

        except Exception as e:
            self.logger.error(f"Error actualizando historial: {e}")

    def _cleanup_old_models(self, model_files: List[Path], keep_count: int = 3):
        """Limpia modelos antiguos manteniendo solo los más recientes"""
        if len(model_files) <= keep_count:
            return

        # Ordenar por fecha de modificación (más reciente primero)
        sorted_models = sorted(
            model_files, key=lambda x: x.stat().st_mtime, reverse=True
        )

        # Eliminar los más antiguos
        to_delete = sorted_models[keep_count:]
        deleted_count = 0

        for model_file in to_delete:
            try:
                model_file.unlink()
                deleted_count += 1
                self.logger.info(f"Modelo antiguo eliminado: {model_file.name}")
            except Exception as e:
                self.logger.error(f"Error eliminando {model_file.name}: {e}")

        print(f"✅ Eliminados {deleted_count} modelos antiguos")
        print(f"📦 Mantenidos {keep_count} modelos más recientes")
