#!/usr/bin/env python3
"""
data_preprocessor.py - Preprocesamiento avanzado con distribución inteligente
Versión 3.1 - CORREGIDO: Estructura Kohya_ss automática con formato {repeats}_{client_id}
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from config import CONFIG
from utils import (
    PipelineLogger,
    ProgressTracker,
    safe_copy_file,
    load_json_safe,
    save_json_safe,
)

try:
    import rawpy
    import imageio

    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False


class DataPreprocessor:
    """Preprocesador de datos con distribución inteligente y soporte RAW completo"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.logger = PipelineLogger("DataPreprocessor", self.config.logs_dir)

        self.mj_parameters = [
            "seed",
            "version",
            "quality",
            "stylize",
            "chaos",
            "aspect_ratio",
            "style",
            "omni_weight",
            "variety",
            "weirdness",
        ]

        self.stats = {
            "mj_imported": 0,
            "real_imported": 0,
            "raw_conversions": 0,
            "metadata_extracted": 0,
        }

    def process_mj_images(
        self, client_id: str, source_dir: str, clients_dir: Path
    ) -> bool:
        """Procesa imágenes MJ con captura completa de metadata"""
        client_path = clients_dir / client_id
        raw_mj_dir = client_path / "raw_mj"
        metadata_dir = client_path / "metadata"

        raw_mj_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Procesando imágenes MidJourney para cliente: {client_id}")
        self.logger.info(f"Origen: {source_dir}")
        self.logger.info(f"Destino: {raw_mj_dir}")

        image_files = self._get_image_files(source_dir)

        if not image_files:
            self.logger.error(f"No se encontraron imágenes en {source_dir}")
            return False

        self.logger.info(f"Encontradas {len(image_files)} imágenes únicas")

        image_groups = self._group_mj_images_by_uuid(image_files)
        self.logger.info(f"Detectados {len(image_groups)} grupos de imágenes")

        client_config = self._load_client_config(client_path)
        if not client_config.get("prompt_maestro"):
            prompt_maestro = self._capture_master_prompt()
            client_config["prompt_maestro"] = prompt_maestro
            self._save_client_config(client_path, client_config)
            self.logger.info("Prompt maestro guardado")

        group_prompts = self._capture_group_prompts(image_groups)

        metadata_mapping = {}
        tracker = ProgressTracker(len(image_files), "Importando imágenes MJ")

        copied_count = 0
        for group_uuid, group_files in image_groups.items():
            group_prompt = group_prompts.get(group_uuid, "")

            for i, image_file in enumerate(group_files):
                mj_metadata = self._extract_complete_mj_metadata(
                    image_file.name, group_prompt, client_config.get("omni_weight", 160)
                )

                new_name = self._generate_mj_filename(client_id, mj_metadata, i + 1)

                src = image_file
                dst = raw_mj_dir / new_name

                if safe_copy_file(src, dst, self.logger):
                    metadata_mapping[new_name] = mj_metadata
                    copied_count += 1
                    self.stats["mj_imported"] += 1

                tracker.update(status=f"Procesada: {new_name}")

        tracker.finish("Importación MJ completada")

        mapping_file = metadata_dir / "mj_metadata_mapping.json"
        save_json_safe(metadata_mapping, mapping_file, self.logger)

        prompts_analysis = self._analyze_prompt_patterns(metadata_mapping)
        analysis_file = metadata_dir / "mj_prompts_analysis.json"
        save_json_safe(prompts_analysis, analysis_file, self.logger)

        self.logger.info(f"Importación MJ completada:")
        self.logger.info(f"  Imágenes importadas: {copied_count}")
        self.logger.info(f"  Grupos procesados: {len(image_groups)}")
        self.logger.info(f"  Metadata guardada: {mapping_file}")

        return True

    def process_real_images(
        self, client_id: str, source_dir: str, clients_dir: Path
    ) -> bool:
        """Procesa fotos reales con soporte completo para RAW"""
        client_path = clients_dir / client_id
        raw_real_dir = client_path / "raw_real"
        metadata_dir = client_path / "metadata"
        temp_dir = client_path / "temp_raw_conversion"

        raw_real_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Procesando fotos reales con soporte RAW para cliente: {client_id}"
        )
        self.logger.info(f"Origen: {source_dir}")
        self.logger.info(f"Destino: {raw_real_dir}")

        image_files = self._get_image_files(source_dir)

        if not image_files:
            self.logger.error(f"No se encontraron imágenes soportadas en {source_dir}")
            return False

        metadata_mapping = {}
        tracker = ProgressTracker(len(image_files), "Procesando fotos reales")

        copied_count = 0
        converted_count = 0

        for i, image_file in enumerate(image_files, 1):
            is_raw = image_file.suffix.lower() in [
                ext.lower() for ext in self.config.supported_extensions["raw"]
            ]

            if is_raw and RAW_SUPPORT:
                self.logger.debug(f"Archivo RAW detectado: {image_file.suffix.upper()}")

                temp_jpeg = self._convert_raw_to_temp_jpeg(image_file, temp_dir)
                if temp_jpeg is None:
                    tracker.update(status=f"Error: {image_file.name}")
                    continue

                converted_count += 1
                self.stats["raw_conversions"] += 1
            elif is_raw and not RAW_SUPPORT:
                self.logger.warning(
                    f"Archivo RAW encontrado pero rawpy no disponible: {image_file.name}"
                )
                tracker.update(status=f"Saltado RAW: {image_file.name}")
                continue

            real_metadata = self._analyze_real_image(image_file, i, is_raw=is_raw)

            new_name = self._generate_real_filename(
                client_id, real_metadata, i, is_raw=is_raw
            )

            dst = raw_real_dir / new_name
            if safe_copy_file(image_file, dst, self.logger):
                if is_raw and converted_count > 0:
                    real_metadata["temp_conversion_path"] = str(
                        temp_dir / f"temp_{image_file.stem}.jpg"
                    )

                real_metadata["final_path"] = str(dst)
                metadata_mapping[new_name] = real_metadata
                copied_count += 1
                self.stats["real_imported"] += 1

            tracker.update(status=f"Procesada: {new_name}")

        tracker.finish("Procesamiento fotos reales completado")

        mapping_file = metadata_dir / "real_metadata_mapping.json"
        save_json_safe(metadata_mapping, mapping_file, self.logger)

        real_analysis = self._analyze_real_photos_patterns(metadata_mapping)
        real_analysis["conversion_stats"] = {
            "total_processed": copied_count,
            "raw_conversions": converted_count,
            "standard_files": copied_count - converted_count,
        }

        analysis_file = metadata_dir / "real_photos_analysis.json"
        save_json_safe(real_analysis, analysis_file, self.logger)

        self.logger.info(f"Importación fotos reales completada:")
        self.logger.info(f"  Imágenes procesadas: {copied_count}")
        self.logger.info(f"  Conversiones RAW: {converted_count}")
        self.logger.info(f"  Archivos estándar: {copied_count - converted_count}")

        return True

    def prepare_lora_dataset_kohya(self, client_id: str, clients_dir: Path) -> bool:
        """
        Prepara dataset final para LoRA con estructura KOHYA_SS correcta
        Formato automático: training_data/{repeats}_{client_id}/
        """
        client_path = clients_dir / client_id
        processed_dir = client_path / "processed"
        training_data_parent = client_path / "training_data"
        metadata_dir = client_path / "metadata"

        self.logger.info(
            f"Preparando dataset LoRA con estructura Kohya_ss para: {client_id}"
        )

        if not processed_dir.exists():
            self.logger.error(
                "No hay imágenes procesadas. Ejecuta primero el procesamiento facial."
            )
            return False

        processed_images = list(processed_dir.glob("*.png"))

        if not processed_images:
            self.logger.error("No hay imágenes procesadas disponibles")
            return False

        mj_images = [img for img in processed_images if "_mj_" in img.name]
        real_images = [img for img in processed_images if "_real_" in img.name]

        self.logger.info(f"Imágenes disponibles:")
        self.logger.info(f"  MJ: {len(mj_images)}")
        self.logger.info(f"  Real: {len(real_images)}")
        self.logger.info(f"  Total: {len(processed_images)}")

        avatar_type, distribution = self._determine_avatar_type_and_distribution(
            len(mj_images), len(real_images)
        )

        self.logger.info(f"Análisis de distribución:")
        self.logger.info(f"  Tipo de avatar: {avatar_type}")
        self.logger.info(f"  Distribución objetivo: {distribution['description']}")

        total_available = len(mj_images) + len(real_images)

        if total_available < 30:
            self.logger.warning(
                f"Pocas imágenes disponibles ({total_available}). Recomendado mínimo: 50"
            )

        target_mj_count, target_real_count = self._calculate_distribution(
            len(mj_images), len(real_images), distribution, total_available
        )

        actual_mj_count = min(len(mj_images), target_mj_count)
        actual_real_count = min(len(real_images), target_real_count)

        self.logger.info(f"Distribución final:")
        self.logger.info(
            f"  MJ seleccionadas: {actual_mj_count} ({actual_mj_count/(actual_mj_count+actual_real_count)*100:.1f}%)"
        )
        self.logger.info(
            f"  Real seleccionadas: {actual_real_count} ({actual_real_count/(actual_mj_count+actual_real_count)*100:.1f}%)"
        )

        selected_mj = self._select_best_images(mj_images, actual_mj_count)
        selected_real = self._select_best_images(real_images, actual_real_count)

        # CALCULAR REPETICIONES AUTOMÁTICAMENTE para Kohya_ss
        total_images = actual_mj_count + actual_real_count
        if total_images > 0:
            # Fórmula: ~600 steps efectivos mínimo / total_images
            dataset_repeats = max(10, min(200, 600 // total_images))
        else:
            dataset_repeats = 50  # Fallback

        # CREAR ESTRUCTURA KOHYA_SS con formato correcto: {repeats}_{client_id}
        training_data_subdir = training_data_parent / f"{dataset_repeats}_{client_id}"
        
        # LIMPIAR directorio anterior si existe (evita duplicados al reprocesar)
        if training_data_subdir.exists():
            self.logger.info(f"Limpiando directorio anterior: {training_data_subdir}")
            print(f"🧹 Limpiando dataset anterior...")
            try:
                shutil.rmtree(training_data_subdir)
            except Exception as e:
                self.logger.warning(f"Error limpiando directorio: {e}")
        
        training_data_subdir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Estructura Kohya_ss: {training_data_subdir}")
        self.logger.info(f"Repeticiones automáticas: {dataset_repeats}")
        print(f"🏗️ Estructura Kohya_ss: {training_data_subdir.name}")
        print(f"   Repeticiones del dataset: {dataset_repeats}")

        weights = self._get_weights_for_avatar_type(avatar_type)

        dataset_metadata = {}
        total_to_process = len(selected_mj) + len(selected_real)
        tracker = ProgressTracker(total_to_process, "Preparando dataset LoRA Kohya_ss")

        for i, img_path in enumerate(selected_mj, 1):
            dst_name = f"{client_id}_mj_{i:03d}.png"
            dst_path = training_data_subdir / dst_name

            if safe_copy_file(img_path, dst_path, self.logger):
                caption = self._generate_mj_caption(client_id, img_path, metadata_dir)
                caption_file = training_data_subdir / f"{client_id}_mj_{i:03d}.txt"

                try:
                    with open(caption_file, "w", encoding="utf-8") as f:
                        f.write(caption)

                    dataset_metadata[dst_name] = {
                        "type": "mj",
                        "weight": weights["mj_weight"],
                        "caption": caption,
                        "original_path": str(img_path),
                        "kohya_path": str(dst_path),
                        "kohya_caption_path": str(caption_file),
                    }
                except Exception as e:
                    self.logger.error(f"Error escribiendo caption {caption_file}: {e}")

            tracker.update(status=f"MJ: {dst_name}")

        for i, img_path in enumerate(selected_real, 1):
            dst_name = f"{client_id}_real_{i:03d}.png"
            dst_path = training_data_subdir / dst_name

            if safe_copy_file(img_path, dst_path, self.logger):
                caption = self._generate_real_caption(client_id, img_path, metadata_dir)
                caption_file = training_data_subdir / f"{client_id}_real_{i:03d}.txt"

                try:
                    with open(caption_file, "w", encoding="utf-8") as f:
                        f.write(caption)

                    dataset_metadata[dst_name] = {
                        "type": "real",
                        "weight": weights["real_weight"],
                        "caption": caption,
                        "original_path": str(img_path),
                        "kohya_path": str(dst_path),
                        "kohya_caption_path": str(caption_file),
                    }
                except Exception as e:
                    self.logger.error(f"Error escribiendo caption {caption_file}: {e}")

            tracker.update(status=f"Real: {dst_name}")

        tracker.finish("Dataset LoRA Kohya_ss completado")

        # CREAR ARCHIVO DE CONFIGURACIÓN PARA KOHYA_SS
        self._create_kohya_dataset_config(
            training_data_subdir, 
            client_id, 
            dataset_repeats,
            len(dataset_metadata)
        )

        dataset_info = {
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "avatar_type": avatar_type,
            "distribution_strategy": distribution,
            "total_images": len(dataset_metadata),
            "mj_images": actual_mj_count,
            "real_images": actual_real_count,
            "balance_ratio": f"{actual_mj_count/(actual_mj_count+actual_real_count)*100:.1f}% MJ / {actual_real_count/(actual_mj_count+actual_real_count)*100:.1f}% Real",
            "weight_config": weights,
            "kohya_structure": {
                "training_data_parent": str(training_data_parent),
                "training_data_subdir": str(training_data_subdir),
                "subdir_name": training_data_subdir.name,
                "repeats": dataset_repeats,
                "structure_version": "kohya_compatible_v2",
                "total_files": len(dataset_metadata) * 2,
            },
            "recommended_training": {
                "total_steps": min(3500, len(dataset_metadata) * 30),
                "batch_size": 1,
                "learning_rate": 0.0001 if len(dataset_metadata) < 100 else 0.00008,
                "dataset_repeats": dataset_repeats,
            },
        }

        dataset_config_file = metadata_dir / "lora_dataset_config_kohya.json"
        save_json_safe(dataset_info, dataset_config_file, self.logger)

        metadata_file = metadata_dir / "lora_dataset_metadata_kohya.json"
        save_json_safe(dataset_metadata, metadata_file, self.logger)

        trainer_info_file = training_data_subdir / "dataset_info.json"
        trainer_info = {
            "client_id": client_id,
            "total_images": len(dataset_metadata),
            "structure_type": "kohya_compatible_v2",
            "repeats": dataset_repeats,
            "created_date": datetime.now().isoformat(),
        }
        save_json_safe(trainer_info, trainer_info_file, self.logger)

        self.logger.info(f"Dataset LoRA Kohya_ss preparado exitosamente:")
        self.logger.info(f"  Avatar tipo: {avatar_type}")
        self.logger.info(f"  Total imágenes: {len(dataset_metadata)}")
        self.logger.info(f"  Balance: {dataset_info['balance_ratio']}")
        self.logger.info(f"  Captions generados: {len(dataset_metadata)}")
        self.logger.info(f"  Estructura Kohya_ss: {training_data_subdir}")
        self.logger.info(f"  Formato: {training_data_subdir.name}")

        return True

    # === MÉTODOS AUXILIARES (sin cambios) ===

    def _get_image_files(self, source_dir: str) -> List[Path]:
        """Obtiene archivos de imagen evitando duplicados y soportando RAW"""
        source_path = Path(source_dir)

        all_extensions = (
            self.config.supported_extensions["standard"]
            + self.config.supported_extensions["raw"]
        )

        unique_files = []
        seen_paths = set()

        for file_path in source_path.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in [ext.lower() for ext in all_extensions]:
                    resolved_path = file_path.resolve()
                    if resolved_path not in seen_paths:
                        seen_paths.add(resolved_path)
                        unique_files.append(file_path)

        raw_files = [
            f
            for f in unique_files
            if f.suffix.lower()
            in [ext.lower() for ext in self.config.supported_extensions["raw"]]
        ]
        standard_files = [
            f
            for f in unique_files
            if f.suffix.lower()
            in [ext.lower() for ext in self.config.supported_extensions["standard"]]
        ]

        self.logger.debug(f"Archivos encontrados: {len(unique_files)} total")
        if raw_files:
            self.logger.info(f"Archivos RAW: {len(raw_files)}")
        if standard_files:
            self.logger.info(f"Archivos estándar: {len(standard_files)}")

        return sorted(unique_files)

    def _convert_raw_to_temp_jpeg(
        self, raw_path: Path, temp_dir: Path
    ) -> Optional[Path]:
        """Convierte archivo RAW a JPEG temporal para procesamiento"""
        if not RAW_SUPPORT:
            self.logger.error("rawpy no está disponible")
            return None

        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"temp_{raw_path.stem}.jpg"
        temp_path = temp_dir / temp_name

        try:
            with rawpy.imread(str(raw_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8,
                    bright=1.0,
                )

            imageio.imwrite(str(temp_path), rgb, quality=98)
            self.logger.debug(f"Conversión RAW exitosa: {temp_path}")
            return temp_path

        except Exception as e:
            self.logger.warning(f"Error convirtiendo {raw_path.name}: {str(e)}")

            try:
                with rawpy.imread(str(raw_path)) as raw:
                    rgb = raw.postprocess()

                imageio.imwrite(str(temp_path), rgb, quality=95)
                self.logger.info(
                    f"Conversión RAW exitosa con configuración básica: {temp_path}"
                )
                return temp_path

            except Exception as e2:
                self.logger.error(f"Conversión RAW falló completamente: {str(e2)}")
                return None

    def _determine_avatar_type_and_distribution(
        self, mj_count: int, real_count: int
    ) -> Tuple[str, Dict]:
        """Determina el tipo de avatar y distribución según configuración y datos disponibles"""

        if real_count == 0:
            avatar_type = "synthetic"
            distribution = {
                "description": "100% MidJourney (solo sintético)",
                "mj_ratio": 1.0,
                "real_ratio": 0.0,
            }
        elif mj_count == 0:
            avatar_type = "real_only"
            distribution = {
                "description": "100% Fotos Reales (solo real)",
                "mj_ratio": 0.0,
                "real_ratio": 1.0,
            }
        elif real_count >= mj_count * self.config.dataset_dist.real_dominant_threshold:
            avatar_type = "real_dominant"
            distribution = {
                "description": f"{self.config.dataset_dist.avatar_real_real_ratio*100:.0f}% Real / {self.config.dataset_dist.avatar_real_mj_ratio*100:.0f}% MJ (avatar real)",
                "mj_ratio": self.config.dataset_dist.avatar_real_mj_ratio,
                "real_ratio": self.config.dataset_dist.avatar_real_real_ratio,
            }
        else:
            avatar_type = "balanced_synthetic"
            distribution = {
                "description": f"{self.config.dataset_dist.synthetic_ratio*100:.0f}% MJ / {self.config.dataset_dist.real_ratio*100:.0f}% Real (avatar sintético balanceado)",
                "mj_ratio": self.config.dataset_dist.synthetic_ratio,
                "real_ratio": self.config.dataset_dist.real_ratio,
            }

        return avatar_type, distribution

    def _calculate_distribution(
        self,
        mj_available: int,
        real_available: int,
        distribution: Dict,
        total_target: int,
    ) -> Tuple[int, int]:
        """Calcula la distribución exacta de imágenes"""

        target_mj = int(total_target * distribution["mj_ratio"])
        target_real = int(total_target * distribution["real_ratio"])

        if target_mj > mj_available:
            target_mj = mj_available
            remaining_slots = total_target - target_mj
            target_real = min(real_available, remaining_slots)

        if target_real > real_available:
            target_real = real_available
            remaining_slots = total_target - target_real
            target_mj = min(mj_available, remaining_slots)

        return target_mj, target_real

    def _get_weights_for_avatar_type(self, avatar_type: str) -> Dict[str, Any]:
        """Devuelve pesos de entrenamiento según el tipo de avatar"""

        weight_configs = {
            "synthetic": {
                "mj_weight": 1.0,
                "real_weight": 0.0,
                "description": "MJ peso completo",
            },
            "real_only": {
                "mj_weight": 0.0,
                "real_weight": 1.0,
                "description": "Real peso completo",
            },
            "real_dominant": {
                "mj_weight": 0.8,
                "real_weight": 1.0,
                "description": "Real dominante, MJ soporte",
            },
            "balanced_synthetic": {
                "mj_weight": 1.0,
                "real_weight": 0.3,
                "description": "MJ dominante, Real soporte",
            },
        }

        return weight_configs.get(avatar_type, weight_configs["balanced_synthetic"])

    def _group_mj_images_by_uuid(
        self, image_files: List[Path]
    ) -> Dict[str, List[Path]]:
        """Agrupa imágenes MJ por UUID (4 variaciones por grupo)"""
        uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
        groups = defaultdict(list)

        for image_file in image_files:
            match = re.search(uuid_pattern, image_file.name)
            if match:
                uuid = match.group(1)
                groups[uuid].append(image_file)
            else:
                groups["no_uuid"].append(image_file)

        return dict(groups)

    def _capture_master_prompt(self) -> str:
        """Captura el prompt maestro del cliente interactivamente"""
        print(f"\n📝 CAPTURA DE PROMPT MAESTRO")
        print("=" * 40)
        print("Este es el prompt base exitoso que usaste para el avatar experimental.")
        print("Será usado como referencia para generar captions ricos.")
        print()

        while True:
            prompt = input("Ingresa el prompt maestro: ").strip()
            if prompt:
                print(f"✅ Prompt maestro capturado ({len(prompt)} caracteres)")
                return prompt
            else:
                print("❌ El prompt no puede estar vacío")

    def _capture_group_prompts(
        self, image_groups: Dict[str, List[Path]]
    ) -> Dict[str, str]:
        """Captura prompts específicos para cada grupo de imágenes"""
        print(f"\n📝 CAPTURA DE PROMPTS POR GRUPO")
        print("=" * 40)
        print("Ingresa el prompt específico usado para cada grupo de 4 imágenes.")
        print()

        group_prompts = {}
        for i, (group_uuid, group_files) in enumerate(image_groups.items(), 1):
            if group_uuid == "no_uuid":
                print(f"📷 Grupo {i}: Imágenes sin UUID ({len(group_files)} archivos)")
            else:
                print(
                    f"📷 Grupo {i}: UUID {group_uuid[:8]}... ({len(group_files)} archivos)"
                )

            print(f"   Ejemplo: {group_files[0].name[:50]}...")

            while True:
                prompt = input(f"Prompt para grupo {i}: ").strip()
                if prompt:
                    group_prompts[group_uuid] = prompt
                    print(f"   ✅ Guardado ({len(prompt)} caracteres)")
                    break
                else:
                    skip = input(
                        "   ⚠️ Prompt vacío. ¿Saltar este grupo? (s/n): "
                    ).lower()
                    if skip.startswith("s"):
                        group_prompts[group_uuid] = ""
                        break

        return group_prompts

    def _extract_complete_mj_metadata(
        self, filename: str, group_prompt: str, omni_weight: int = 160
    ) -> Dict[str, Any]:
        """Extrae metadata completa de archivo MJ"""
        patterns = {
            "user_id": r"^u(\d+)_",
            "uuid": r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
            "variant": r"_(\d+)\.png$",
            "seed": r"seed[_-]?(\d+)",
            "version": r"--v\s*(\d+)",
            "quality": r"--q\s*(\d+)",
            "stylize": r"--s\s*(\d+)",
            "chaos": r"--chaos\s*(\d+)",
            "aspect_ratio": r"--ar\s*([\d:]+)",
            "style": r"--style\s*(\w+)",
            "omni_weight": r"--ow\s*(\d+)",
            "variety": r"--variety\s*(\d+)",
            "weirdness": r"--weird\s*(\d+)",
        }

        metadata = {
            "original_filename": filename,
            "group_prompt": group_prompt,
            "import_date": datetime.now().isoformat(),
        }

        for param, pattern in patterns.items():
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                if param in [
                    "version",
                    "quality",
                    "stylize",
                    "chaos",
                    "omni_weight",
                    "variety",
                    "weirdness",
                ]:
                    metadata[param] = int(match.group(1))
                else:
                    metadata[param] = match.group(1)
            elif group_prompt and param not in ["user_id", "uuid", "variant"]:
                match = re.search(pattern, group_prompt, re.IGNORECASE)
                if match:
                    if param in [
                        "version",
                        "quality",
                        "stylize",
                        "chaos",
                        "omni_weight",
                        "variety",
                        "weirdness",
                    ]:
                        metadata[param] = int(match.group(1))
                    else:
                        metadata[param] = match.group(1)

        defaults = {
            "version": 7,
            "quality": 5,
            "style": "raw",
            "omni_weight": omni_weight,
            "stylize": 800,
            "chaos": 0,
            "aspect_ratio": "3:4",
        }

        for param, default_value in defaults.items():
            if param not in metadata:
                metadata[param] = default_value

        metadata["detected_features"] = self._detect_prompt_features(group_prompt)

        if group_prompt:
            clean_prompt = re.sub(r"--\w+\s*\d*", "", group_prompt).strip()
            metadata["clean_prompt"] = clean_prompt

        self.stats["metadata_extracted"] += 1
        return metadata

    def _detect_prompt_features(self, prompt: str) -> List[str]:
        """Detecta características específicas del prompt"""
        if not prompt:
            return []

        prompt_lower = prompt.lower()
        detected = []

        lighting_features = {
            "studio_lighting": ["studio lighting", "professional lighting"],
            "natural_lighting": ["natural light", "window light", "daylight"],
            "dramatic_lighting": ["dramatic lighting", "rembrandt lighting"],
            "soft_lighting": ["soft light", "gentle light"],
            "cinematic_lighting": ["cinematic lighting", "film lighting"],
        }

        expression_features = {
            "confident": ["confident", "confidence", "assertive"],
            "serene": ["serene", "calm", "peaceful"],
            "trustworthy": ["trustworthy", "reliable", "honest"],
            "wise": ["wise", "wisdom", "sage"],
            "professional": ["professional", "corporate", "business"],
        }

        physical_features = {
            "clean_shaven": ["clean-shaven", "clean shaven"],
            "stubble": ["stubble", "short beard", "facial hair"],
            "well_groomed": ["well-groomed", "well groomed", "neat"],
            "refined": ["refined", "polished", "sophisticated"],
        }

        all_features = {**lighting_features, **expression_features, **physical_features}

        for feature_name, keywords in all_features.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected.append(feature_name)

        return detected

    def _generate_mj_filename(self, client_id: str, metadata: Dict, index: int) -> str:
        """Genera nombre de archivo preservando metadata importante"""
        uuid_short = metadata.get("uuid", "no_uuid")
        if uuid_short != "no_uuid" and len(uuid_short) > 8:
            uuid_short = uuid_short.split("-")[-1]

        features = metadata.get("detected_features", [])
        features_str = "-".join(features[:2]) if features else "default"

        version = metadata.get("version", 7)
        style = metadata.get("style", "raw")

        filename = f"{client_id}_mj_{index:03d}_{uuid_short}_{features_str}_v{version}_{style}.png"
        return filename

    def _analyze_prompt_patterns(self, metadata_mapping: Dict) -> Dict[str, Any]:
        """Analiza patrones en los prompts para insights"""
        analysis = {
            "total_images": len(metadata_mapping),
            "feature_frequency": {},
            "parameter_distribution": {},
            "prompt_analysis": {},
            "generation_date": datetime.now().isoformat(),
        }

        all_features = []
        for metadata in metadata_mapping.values():
            features = metadata.get("detected_features", [])
            all_features.extend(features)

        analysis["feature_frequency"] = dict(Counter(all_features))

        for param in self.mj_parameters:
            values = []
            for metadata in metadata_mapping.values():
                if param in metadata:
                    values.append(metadata[param])

            if values:
                if isinstance(values[0], (int, float)):
                    analysis["parameter_distribution"][param] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "unique_values": list(set(values)),
                    }
                else:
                    analysis["parameter_distribution"][param] = {
                        "unique_values": list(set(values)),
                        "frequency": dict(Counter(values)),
                    }

        return analysis

    def _analyze_real_image(
        self, image_file: Path, index: int, is_raw: bool = False
    ) -> Dict[str, Any]:
        """Analiza automáticamente características de foto real"""
        filename = image_file.name.lower()
        detected_features = []

        if any(word in filename for word in ["studio", "professional", "pro"]):
            detected_features.append("studio_lighting")
        elif any(word in filename for word in ["natural", "window", "outdoor"]):
            detected_features.append("natural_lighting")

        if any(word in filename for word in ["hq", "high", "quality"]):
            detected_features.append("high_quality")
        elif any(word in filename for word in ["casual", "phone", "mobile"]):
            detected_features.append("casual_quality")

        if is_raw:
            detected_features.extend(
                ["raw_format", "high_dynamic_range", "professional_capture"]
            )

        session_group = f"session_{(index-1)//10 + 1:02d}"

        metadata = {
            "original_filename": image_file.name,
            "import_date": datetime.now().isoformat(),
            "session_group": session_group,
            "detected_features": detected_features,
            "image_type": "real_photo",
            "is_raw_format": is_raw,
            "original_format": image_file.suffix.upper(),
            "purpose": "deepfacelive_compatibility",
            "file_size_mb": image_file.stat().st_size / (1024 * 1024),
        }

        return metadata

    def _generate_real_filename(
        self, client_id: str, metadata: Dict, index: int, is_raw: bool = False
    ) -> str:
        """Genera nombre para foto real preservando información de formato"""
        session = metadata.get("session_group", f"session_{index:02d}")
        features = metadata.get("detected_features", [])
        features_str = "-".join(features[:2]) if features else "standard"

        original_ext = metadata.get("original_format", ".JPG").lower()
        format_indicator = "raw" if is_raw else "std"

        filename = f"{client_id}_real_{index:03d}_{session}_{format_indicator}_{features_str}{original_ext}"
        return filename

    def _analyze_real_photos_patterns(self, metadata_mapping: Dict) -> Dict[str, Any]:
        """Analiza patrones en fotos reales"""
        analysis = {
            "total_images": len(metadata_mapping),
            "feature_frequency": {},
            "session_distribution": {},
            "format_distribution": {},
            "avg_file_size_mb": 0,
            "analysis_date": datetime.now().isoformat(),
        }

        all_features = []
        session_counts = []
        format_counts = []
        file_sizes = []

        for metadata in metadata_mapping.values():
            features = metadata.get("detected_features", [])
            all_features.extend(features)

            session = metadata.get("session_group", "unknown")
            session_counts.append(session)

            original_format = metadata.get("original_format", "unknown")
            format_counts.append(original_format)

            size = metadata.get("file_size_mb", 0)
            if size > 0:
                file_sizes.append(size)

        analysis["feature_frequency"] = dict(Counter(all_features))
        analysis["session_distribution"] = dict(Counter(session_counts))
        analysis["format_distribution"] = dict(Counter(format_counts))

        if file_sizes:
            analysis["avg_file_size_mb"] = sum(file_sizes) / len(file_sizes)

        return analysis

    def _select_best_images(
        self, image_list: List[Path], target_count: int
    ) -> List[Path]:
        """Selecciona las mejores imágenes basado en criterios de calidad"""
        if len(image_list) <= target_count:
            return image_list

        sorted_images = sorted(image_list, key=lambda x: x.stat().st_size, reverse=True)
        return sorted_images[:target_count]

    def _generate_mj_caption(
        self, client_id: str, image_path: Path, metadata_dir: Path
    ) -> str:
        """Genera caption rico para imagen MJ usando metadata preservada"""
        mj_metadata_file = metadata_dir / "mj_metadata_mapping.json"
        mj_metadata = load_json_safe(mj_metadata_file, {})

        for original_name, metadata in mj_metadata.items():
            if original_name in image_path.name or any(
                part in image_path.name for part in original_name.split("_")
            ):
                prompt_base = metadata.get("clean_prompt", f"portrait of {client_id}")
                features = metadata.get("detected_features", [])

                caption_parts = [prompt_base]
                if features:
                    caption_parts.append(", ".join(features))

                caption_parts.extend(
                    ["detailed face", "high quality", "professional photography"]
                )
                return ", ".join(caption_parts)

        return f"portrait of {client_id}, detailed face, high quality photography, professional headshot"

    def _generate_real_caption(
        self, client_id: str, image_path: Path, metadata_dir: Path
    ) -> str:
        """Genera caption para foto real enfocado en compatibilidad deepfacelive"""
        real_metadata_file = metadata_dir / "real_metadata_mapping.json"
        real_metadata = load_json_safe(real_metadata_file, {})

        base_caption = f"reference photo of {client_id}, natural facial structure, geometric anchor points, detailed facial mapping"

        for original_name, metadata in real_metadata.items():
            if original_name in image_path.name:
                features = metadata.get("detected_features", [])
                if features:
                    feature_text = ", ".join(features)
                    return f"{base_caption}, {feature_text}, compatible for face swap technology"

        return f"{base_caption}, compatible for face swap technology"

    def _load_client_config(self, client_path: Path) -> Dict[str, Any]:
        """Carga configuración del cliente"""
        config_file = client_path / "metadata" / "client_config.json"
        return load_json_safe(config_file, {})

    def _save_client_config(self, client_path: Path, config: Dict[str, Any]):
        """Guarda configuración del cliente"""
        config_file = client_path / "metadata" / "client_config.json"
        save_json_safe(config, config_file, self.logger)

    def _create_kohya_dataset_config(
        self, 
        dataset_dir: Path, 
        client_id: str, 
        repeats: int,
        num_images: int
    ):
        """
        Crea archivo de configuración para Kohya_ss
        Este archivo es CRÍTICO para que Kohya_ss detecte las imágenes
        """
        config_content = f"""# Kohya_ss Dataset Configuration
# Auto-generated by Avatar Pipeline

[[datasets]]
resolution = 768
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{dataset_dir.as_posix()}" 
  num_repeats = {repeats}
  class_tokens = "{client_id}"
  caption_extension = ".txt"
  keep_tokens = 1
  shuffle_caption = true
  
  # Total: {num_images} images × {repeats} repeats = {num_images * repeats} steps per epoch
"""
        
        # Guardar en el directorio PADRE (training_data/)
        config_file = dataset_dir.parent / "dataset_config.toml"
        
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(config_content)
            
            self.logger.info(f"Archivo de configuración Kohya_ss creado: {config_file}")
            print(f"📄 Config Kohya_ss: {config_file.name}")
            
        except Exception as e:
            self.logger.error(f"Error creando config Kohya_ss: {e}")