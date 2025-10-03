#!/usr/bin/env python3
"""
config.py - Configuración centralizada del sistema Avatar Pipeline
Versión 3.2 - CORREGIDO: Preset Producción Profesional optimizado
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class QualityControlConfig:
    """Configuración de control de calidad facial"""

    face_confidence_threshold: float = 0.85
    face_padding_factor: float = 1.6
    min_file_size_kb: int = 200
    max_file_size_mb: int = 5
    min_brightness: int = 40
    max_brightness: int = 220
    min_contrast: int = 25
    blur_threshold: int = 100

    # Configuraciones específicas para diferentes fuentes
    raw_adjustments: Dict[str, float] = field(
        default_factory=lambda: {
            "min_brightness": 0.7,
            "max_brightness": 1.1,
            "min_contrast": 0.5,
            "blur_threshold": 0.08,
        }
    )


@dataclass
class GPUProfile:
    """Perfil de configuración para GPU específica"""

    name: str
    vram_gb_min: float
    vram_gb_max: float
    network_dim: int
    network_alpha: int
    batch_size: int
    resolution: int
    optimizer: str
    conv_lora: bool
    gradient_accumulation_steps: int
    steps_per_hour_estimate: int
    memory_optimizations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetDistribution:
    """Configuración de distribución de dataset"""

    synthetic_ratio: float = 0.9
    real_ratio: float = 0.1
    real_dominant_threshold: float = 2.0
    avatar_real_mj_ratio: float = 0.3
    avatar_real_real_ratio: float = 0.7


class AvatarPipelineConfig:
    """Configuración principal del sistema Avatar Pipeline"""

    def __init__(self):
        self.base_dir = Path(".")
        self.clients_dir = self.base_dir / "clients"
        self.training_dir = self.base_dir / "training"
        self.temp_dir = self.base_dir / "temp"
        self.logs_dir = self.base_dir / "logs"

        self.qc = QualityControlConfig()
        self.dataset_dist = DatasetDistribution()

        self.supported_extensions = {
            "standard": [".png", ".jpg", ".jpeg", ".tiff", ".tif"],
            "raw": [".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"],
        }

        self.gpu_profiles = self._init_gpu_profiles()
        self.training_presets = self._init_training_presets()

        self.logging_config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
        }

    def _init_gpu_profiles(self) -> Dict[str, GPUProfile]:
        """Inicializa perfiles de GPU - CORREGIDO: GTX 1650 usa 768x768"""
        return {
            "gtx_1650": GPUProfile(
                name="GTX 1650 4GB (Optimizado Extremo)",
                vram_gb_min=3.5,
                vram_gb_max=4.5,
                network_dim=32,
                network_alpha=16,
                batch_size=1,
                resolution=768,  # ⭐ CRÍTICO: 768 NO 1024
                optimizer="AdamW8bit",
                conv_lora=False,
                gradient_accumulation_steps=8,
                steps_per_hour_estimate=150,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": True,
                    "medvram": True,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": True,
                    "cpu_offload": False,
                },
            ),
            "rtx_3050": GPUProfile(
                name="RTX 3050 8GB",
                vram_gb_min=7.5,
                vram_gb_max=8.5,
                network_dim=64,
                network_alpha=32,
                batch_size=1,
                resolution=1024,
                optimizer="AdamW8bit",
                conv_lora=True,
                gradient_accumulation_steps=3,
                steps_per_hour_estimate=400,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": False,
                    "medvram": False,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": False,
                    "cpu_offload": False,
                },
            ),
            "rtx_3060": GPUProfile(
                name="RTX 3060 8GB",
                vram_gb_min=7.5,
                vram_gb_max=8.5,
                network_dim=96,
                network_alpha=48,
                batch_size=1,
                resolution=1024,
                optimizer="AdamW8bit",
                conv_lora=True,
                gradient_accumulation_steps=2,
                steps_per_hour_estimate=450,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": False,
                    "medvram": False,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": False,
                    "cpu_offload": False,
                },
            ),
            "rtx_4060": GPUProfile(
                name="RTX 4060 8GB",
                vram_gb_min=7.5,
                vram_gb_max=8.5,
                network_dim=128,
                network_alpha=64,
                batch_size=1,
                resolution=1024,
                optimizer="AdamW8bit",
                conv_lora=True,
                gradient_accumulation_steps=2,
                steps_per_hour_estimate=500,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": False,
                    "medvram": False,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": False,
                    "cpu_offload": False,
                },
            ),
            "rtx_4060_ti": GPUProfile(
                name="RTX 4060 Ti 16GB",
                vram_gb_min=15.0,
                vram_gb_max=17.0,
                network_dim=128,
                network_alpha=64,
                batch_size=2,
                resolution=1024,
                optimizer="AdamW",
                conv_lora=True,
                gradient_accumulation_steps=1,
                steps_per_hour_estimate=600,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": False,
                    "medvram": False,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": False,
                    "cpu_offload": False,
                },
            ),
            "low_end": GPUProfile(
                name="GPU 4GB (GTX 1650, RTX 3050 4GB)",
                vram_gb_min=3.5,
                vram_gb_max=5.0,
                network_dim=32,
                network_alpha=16,
                batch_size=1,
                resolution=768,
                optimizer="AdamW8bit",
                conv_lora=False,
                gradient_accumulation_steps=8,
                steps_per_hour_estimate=150,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": True,
                    "medvram": True,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": True,
                    "cpu_offload": False,
                },
            ),
            "high_end": GPUProfile(
                name="GPU 12GB+ (RTX 3080, 4070, 4080)",
                vram_gb_min=11.0,
                vram_gb_max=24.0,
                network_dim=128,
                network_alpha=64,
                batch_size=2,
                resolution=1024,
                optimizer="AdamW",
                conv_lora=True,
                gradient_accumulation_steps=1,
                steps_per_hour_estimate=700,
                memory_optimizations={
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True,
                    "cache_latents": True,
                    "cache_text_encoder_outputs": True,
                    "lowvram": False,
                    "medvram": False,
                    "xformers_memory_efficient_attention": True,
                    "attention_slicing": False,
                    "cpu_offload": False,
                },
            ),
        }

    def _init_training_presets(self) -> Dict[str, Dict[str, Any]]:
        """Presets de entrenamiento - OPTIMIZADO: Producción Profesional mejorado"""
        return {
            "gtx1650_quick": {
                "name": "GTX 1650 - Rápido (3-4 horas)",
                "max_train_steps": 500,  # Reducido para testing
                "learning_rate": 0.00008,  # MÁS BAJO - evita NaN
                "dataset_repeats_multiplier": 250,
                "save_every_n_steps": 125,
                "description": "Pruebas ultrarrápidas GTX 1650 @ 768x768 (parámetros estables)",
            },
            "gtx1650_balanced": {
                "name": "GTX 1650 - Equilibrado (6-8 horas)",
                "max_train_steps": 1000,
                "learning_rate": 0.00008,  # MÁS BAJO - evita NaN
                "dataset_repeats_multiplier": 200,
                "save_every_n_steps": 250,
                "description": "Balance calidad/tiempo GTX 1650 @ 768x768 (estable)",
            },
            "gtx1650_quality": {
                "name": "GTX 1650 - Alta Calidad (10-12 horas)",
                "max_train_steps": 1500,
                "learning_rate": 0.00006,  # AÚN MÁS BAJO para convergencia suave
                "dataset_repeats_multiplier": 150,
                "save_every_n_steps": 300,
                "description": "Máxima calidad GTX 1650 @ 768x768 (ultra estable)",
            },
            "quick": {
                "name": "Entrenamiento Rápido",
                "max_train_steps": 1500,
                "learning_rate": 0.00012,
                "dataset_repeats_multiplier": 200,
                "save_every_n_steps": 500,
                "description": "Para pruebas rápidas y validación",
            },
            "balanced": {
                "name": "Entrenamiento Equilibrado",
                "max_train_steps": 2500,
                "learning_rate": 0.0001,
                "dataset_repeats_multiplier": 150,
                "save_every_n_steps": 350,
                "description": "Balance entre calidad y tiempo",
            },
            "quality": {
                "name": "Alta Calidad",
                "max_train_steps": 3500,
                "learning_rate": 0.00008,
                "dataset_repeats_multiplier": 120,
                "save_every_n_steps": 250,
                "description": "Máxima calidad de resultado",
            },
            "production": {
                "name": "Producción Profesional",
                "max_train_steps": 4000,
                "learning_rate": 0.00006,  # ⭐ MÁS BAJO para convergencia suave
                "dataset_repeats_multiplier": 100,
                "save_every_n_steps": 200,
                "description": "⭐ CALIDAD MÁXIMA - Parámetros óptimos para uso profesional",
                # Parámetros avanzados específicos para producción
                "advanced_overrides": {
                    "min_snr_gamma": 5,  # Estabilidad máxima
                    "noise_offset": 0.15,  # Mejor contraste y detalles
                    "lr_scheduler": "cosine_with_restarts",
                    "lr_warmup_ratio": 0.1,  # 10% warmup
                    "weight_decay": 0.005,  # Regularización más fuerte
                },
            },
        }

    def detect_gpu_profile(self) -> Optional[GPUProfile]:
        """Detecta automáticamente el perfil de GPU - PRIORIZA GTX 1650"""
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0).lower()

            # ⭐ PRIORIDAD 1: Detección específica GTX 1650
            if "1650" in gpu_name and vram_gb <= 4.5:
                print(f"⚠️  GTX 1650 detectada - Optimizaciones extremas @ 768x768")
                print(f"   VRAM: {vram_gb:.1f}GB")
                print(f"   Resolution: 768x768 (NO 1024)")
                print(f"   Network dim: 32 (NO 64)")
                return self.gpu_profiles["gtx_1650"]

            # PRIORIDAD 2: Detección específica por nombre
            if "3050" in gpu_name and vram_gb > 7:
                return self.gpu_profiles["rtx_3050"]
            elif "3060" in gpu_name:
                return self.gpu_profiles["rtx_3060"]
            elif "4060 ti" in gpu_name:
                return self.gpu_profiles["rtx_4060_ti"]
            elif "4060" in gpu_name:
                return self.gpu_profiles["rtx_4060"]

            # PRIORIDAD 3: Detección por VRAM
            if vram_gb <= 5.0:
                if vram_gb <= 4.5:
                    return self.gpu_profiles.get(
                        "gtx_1650", self.gpu_profiles["low_end"]
                    )
                else:
                    return self.gpu_profiles["low_end"]
            elif vram_gb >= 11.0:
                return self.gpu_profiles["high_end"]
            else:
                return self.gpu_profiles["rtx_3060"]

        except ImportError:
            return None

    def get_qc_params_for_source(self, is_raw_source: bool = False) -> Dict[str, Any]:
        """Obtiene parámetros QC adaptados por tipo de fuente"""
        base_params = {
            "face_confidence_threshold": self.qc.face_confidence_threshold,
            "face_padding_factor": self.qc.face_padding_factor,
            "min_file_size_kb": self.qc.min_file_size_kb,
            "max_file_size_mb": self.qc.max_file_size_mb,
            "min_brightness": self.qc.min_brightness,
            "max_brightness": self.qc.max_brightness,
            "min_contrast": self.qc.min_contrast,
            "blur_threshold": self.qc.blur_threshold,
        }

        if is_raw_source:
            adj = self.qc.raw_adjustments
            base_params.update(
                {
                    "min_brightness": int(
                        base_params["min_brightness"] * adj["min_brightness"]
                    ),
                    "max_brightness": int(
                        base_params["max_brightness"] * adj["max_brightness"]
                    ),
                    "min_contrast": int(
                        base_params["min_contrast"] * adj["min_contrast"]
                    ),
                    "blur_threshold": int(
                        base_params["blur_threshold"] * adj["blur_threshold"]
                    ),
                    "max_file_size_mb": 8,
                }
            )

        return base_params

    def create_directories(self):
        """Crea la estructura de directorios necesaria"""
        for directory in [
            self.clients_dir,
            self.training_dir,
            self.temp_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_config(self, path: Path):
        """Guarda configuración actual a archivo"""
        config_data = {
            "qc_params": {
                "face_confidence_threshold": self.qc.face_confidence_threshold,
                "face_padding_factor": self.qc.face_padding_factor,
                "min_file_size_kb": self.qc.min_file_size_kb,
                "max_file_size_mb": self.qc.max_file_size_mb,
                "min_brightness": self.qc.min_brightness,
                "max_brightness": self.qc.max_brightness,
                "min_contrast": self.qc.min_contrast,
                "blur_threshold": self.qc.blur_threshold,
                "raw_adjustments": self.qc.raw_adjustments,
            },
            "dataset_distribution": {
                "synthetic_ratio": self.dataset_dist.synthetic_ratio,
                "real_ratio": self.dataset_dist.real_ratio,
                "real_dominant_threshold": self.dataset_dist.real_dominant_threshold,
                "avatar_real_mj_ratio": self.dataset_dist.avatar_real_mj_ratio,
                "avatar_real_real_ratio": self.dataset_dist.avatar_real_real_ratio,
            },
            "supported_extensions": self.supported_extensions,
            "logging_config": self.logging_config,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_config(cls, path: Path) -> "AvatarPipelineConfig":
        """Carga configuración desde archivo"""
        config = cls()

        if not path.exists():
            return config

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "qc_params" in data:
                qc_data = data["qc_params"]
                for key, value in qc_data.items():
                    if hasattr(config.qc, key):
                        setattr(config.qc, key, value)

            if "dataset_distribution" in data:
                dist_data = data["dataset_distribution"]
                for key, value in dist_data.items():
                    if hasattr(config.dataset_dist, key):
                        setattr(config.dataset_dist, key, value)

            if "supported_extensions" in data:
                config.supported_extensions = data["supported_extensions"]

            if "logging_config" in data:
                config.logging_config.update(data["logging_config"])

        except Exception as e:
            print(f"Error cargando configuración: {e}")
            print("Usando configuración por defecto")

        return config


# Instancia global de configuración
CONFIG = AvatarPipelineConfig()
