#!/usr/bin/env python3
"""
data_preprocessor.py
Módulo especializado para preprocesamiento de datos
- Captura completa de metadata MidJourney
- Análisis automático de fotos reales
- Balance de datos 85% MJ / 15% Real
- Generación de captions ricos para entrenamiento LoRA
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd


class DataPreprocessor:
    def __init__(self):
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

    def process_mj_images(self, client_id, source_dir, clients_dir):
        """Procesa imágenes MJ con captura completa de metadata"""
        client_path = clients_dir / client_id
        raw_mj_dir = client_path / "raw_mj"
        metadata_dir = client_path / "metadata"

        # Crear directorios
        raw_mj_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🎨 PROCESANDO IMÁGENES MIDJOURNEY")
        print(f"Origen: {source_dir}")
        print(f"Destino: {raw_mj_dir}")
        print("-" * 50)

        # Obtener archivos de imagen
        extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        image_files = []

        for ext in extensions:
            image_files.extend(Path(source_dir).glob(f"*{ext}"))
            image_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"❌ No se encontraron imágenes en {source_dir}")
            return False

        print(f"📸 Encontradas {len(image_files)} imágenes")

        # Agrupar por UUID (grupos de 4 variaciones)
        image_groups = self.group_mj_images_by_uuid(image_files)

        print(f"🔍 Detectados {len(image_groups)} grupos de imágenes")
        for uuid, files in list(image_groups.items())[:3]:  # Mostrar primeros 3 grupos
            print(f"   Grupo {uuid[:8]}...: {len(files)} imágenes")

        # Capturar prompt maestro para el cliente
        client_config = self.load_client_config(client_path)
        if not client_config.get("prompt_maestro"):
            prompt_maestro = self.capture_master_prompt()
            client_config["prompt_maestro"] = prompt_maestro
            self.save_client_config(client_path, client_config)
            print(f"✅ Prompt maestro guardado")

        # Capturar prompts específicos por grupo
        group_prompts = self.capture_group_prompts(image_groups)

        # Procesar y organizar imágenes
        metadata_mapping = {}
        copied_count = 0

        for group_uuid, group_files in image_groups.items():
            group_prompt = group_prompts.get(group_uuid, "")

            for i, image_file in enumerate(group_files):
                # Extraer metadata completa del filename
                mj_metadata = self.extract_complete_mj_metadata(
                    image_file.name, group_prompt, client_config.get("omni_weight", 160)
                )

                # Generar nuevo nombre preservando información
                new_name = self.generate_mj_filename(client_id, mj_metadata, i + 1)

                # Copiar archivo
                src = image_file
                dst = raw_mj_dir / new_name

                shutil.copy2(str(src), str(dst))

                # Guardar metadata mapping
                metadata_mapping[new_name] = mj_metadata
                copied_count += 1

                if copied_count % 10 == 0:
                    print(
                        f"  📋 Procesadas {copied_count}/{len(image_files)} imágenes..."
                    )

        # Guardar metadata mapping completa
        mapping_file = metadata_dir / "mj_metadata_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(metadata_mapping, f, indent=2)

        # Guardar análisis de prompts
        prompts_analysis = self.analyze_prompt_patterns(metadata_mapping)
        analysis_file = metadata_dir / "mj_prompts_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(prompts_analysis, f, indent=2)

        print(f"\n✅ IMPORTACIÓN MJ COMPLETADA:")
        print(f"   Imágenes importadas: {copied_count}")
        print(f"   Grupos procesados: {len(image_groups)}")
        print(f"   Metadata guardada: {mapping_file}")
        print(f"   Análisis de prompts: {analysis_file}")

        return True

    def group_mj_images_by_uuid(self, image_files):
        """Agrupa imágenes MJ por UUID (4 variaciones por grupo)"""
        uuid_pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"
        groups = defaultdict(list)

        for image_file in image_files:
            match = re.search(uuid_pattern, image_file.name)
            if match:
                uuid = match.group(1)
                groups[uuid].append(image_file)
            else:
                # Imágenes sin UUID van a grupo especial
                groups["no_uuid"].append(image_file)

        return dict(groups)

    def capture_master_prompt(self):
        """Captura el prompt maestro del cliente"""
        print(f"\n📝 CAPTURA DE PROMPT MAESTRO")
        print("=" * 40)
        print("Este es el prompt base exitoso que usaste para el avatar experimental.")
        print("Será usado como referencia para generar captions ricos.")
        print()

        while True:
            prompt = input("Ingresa el prompt maestro: ").strip()
            if prompt:
                print(f"\n✅ Prompt maestro capturado ({len(prompt)} caracteres)")
                return prompt
            else:
                print("❌ El prompt no puede estar vacío")

    def capture_group_prompts(self, image_groups):
        """Captura prompts específicos para cada grupo de imágenes"""
        print(f"\n📝 CAPTURA DE PROMPTS POR GRUPO")
        print("=" * 40)
        print("Ahora ingresa el prompt específico usado para cada grupo de 4 imágenes.")
        print("Puedes copiar/pegar directamente desde tu historial de MJ.")
        print()

        group_prompts = {}

        for i, (group_uuid, group_files) in enumerate(image_groups.items(), 1):
            if group_uuid == "no_uuid":
                print(
                    f"\n📷 Grupo {i}: Imágenes sin UUID ({len(group_files)} archivos)"
                )
                sample_file = group_files[0].name[:50] + "..."
            else:
                print(
                    f"\n📷 Grupo {i}: UUID {group_uuid[:8]}... ({len(group_files)} archivos)"
                )
                sample_file = group_files[0].name[:50] + "..."

            print(f"   Ejemplo: {sample_file}")
            print(
                f"   Archivos: {[f.name[-6:] for f in group_files]}"
            )  # Mostrar terminaciones

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

        print(f"\n✅ Capturados prompts para {len(group_prompts)} grupos")
        return group_prompts

    def extract_complete_mj_metadata(self, filename, group_prompt, omni_weight=160):
        """Extrae metadata completa de archivo MJ con prompt específico"""
        # Patrones para extraer parámetros
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

        # Extraer información básica del filename
        metadata = {
            "original_filename": filename,
            "group_prompt": group_prompt,
            "import_date": datetime.now().isoformat(),
        }

        # Extraer parámetros del filename
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

        # Extraer parámetros del prompt si están disponibles
        if group_prompt:
            for param, pattern in patterns.items():
                if (
                    param not in metadata
                    and param != "user_id"
                    and param != "uuid"
                    and param != "variant"
                ):
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

        # Valores por defecto para parámetros comunes
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

        # Detectar características del prompt
        metadata["detected_features"] = self.detect_prompt_features(group_prompt)

        # Limpiar prompt para análisis
        if group_prompt:
            clean_prompt = re.sub(r"--\w+\s*\d*", "", group_prompt).strip()
            metadata["clean_prompt"] = clean_prompt

        return metadata

    def detect_prompt_features(self, prompt):
        """Detecta características específicas del prompt"""
        if not prompt:
            return []

        prompt_lower = prompt.lower()
        detected = []

        # Características de iluminación
        lighting_features = {
            "studio_lighting": [
                "studio lighting",
                "professional lighting",
                "studio light",
            ],
            "natural_lighting": [
                "natural light",
                "window light",
                "daylight",
                "natural lighting",
            ],
            "dramatic_lighting": [
                "dramatic lighting",
                "rembrandt lighting",
                "dramatic light",
            ],
            "soft_lighting": ["soft light", "soft lighting", "gentle light"],
            "cinematic_lighting": [
                "cinematic lighting",
                "cinematic light",
                "film lighting",
            ],
        }

        # Características de expresión
        expression_features = {
            "confident": ["confident", "confidence", "assertive"],
            "serene": ["serene", "calm", "peaceful", "tranquil"],
            "trustworthy": ["trustworthy", "reliable", "honest"],
            "wise": ["wise", "wisdom", "sage"],
            "professional": ["professional", "corporate", "business"],
        }

        # Características físicas
        physical_features = {
            "clean_shaven": ["clean-shaven", "clean shaven", "cleanshaven"],
            "stubble": ["stubble", "short beard", "facial hair"],
            "well_groomed": ["well-groomed", "well groomed", "neat"],
            "refined": ["refined", "polished", "sophisticated"],
        }

        # Buscar características
        all_features = {**lighting_features, **expression_features, **physical_features}

        for feature_name, keywords in all_features.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected.append(feature_name)

        return detected

    def generate_mj_filename(self, client_id, metadata, index):
        """Genera nombre de archivo preservando metadata importante"""
        uuid_short = metadata.get("uuid", "no_uuid")
        if uuid_short != "no_uuid" and len(uuid_short) > 8:
            uuid_short = uuid_short.split("-")[-1]  # Último segmento del UUID

        features = metadata.get("detected_features", [])
        features_str = "-".join(features[:2]) if features else "default"

        version = metadata.get("version", 7)
        style = metadata.get("style", "raw")

        filename = f"{client_id}_mj_{index:03d}_{uuid_short}_{features_str}_v{version}_{style}.png"

        return filename

    def analyze_prompt_patterns(self, metadata_mapping):
        """Analiza patrones en los prompts para insights"""
        analysis = {
            "total_images": len(metadata_mapping),
            "feature_frequency": Counter(),
            "parameter_distribution": {},
            "prompt_analysis": {},
            "generation_date": datetime.now().isoformat(),
        }

        # Contar frecuencia de características
        all_features = []
        for metadata in metadata_mapping.values():
            features = metadata.get("detected_features", [])
            all_features.extend(features)

        analysis["feature_frequency"] = dict(Counter(all_features))

        # Distribución de parámetros
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

        # Análisis de prompts
        all_prompts = [
            m.get("clean_prompt", "")
            for m in metadata_mapping.values()
            if m.get("clean_prompt")
        ]
        if all_prompts:
            # Palabras más comunes
            all_words = []
            for prompt in all_prompts:
                words = re.findall(r"\b\w+\b", prompt.lower())
                all_words.extend(words)

            analysis["prompt_analysis"] = {
                "total_prompts": len(all_prompts),
                "avg_length": sum(len(p) for p in all_prompts) / len(all_prompts),
                "common_words": dict(Counter(all_words).most_common(20)),
            }

        return analysis

    def process_real_images(self, client_id, source_dir, clients_dir):
        """Procesa fotos reales con análisis automático"""
        client_path = clients_dir / client_id
        raw_real_dir = client_path / "raw_real"
        metadata_dir = client_path / "metadata"

        # Crear directorios
        raw_real_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📷 PROCESANDO FOTOS REALES")
        print(f"Origen: {source_dir}")
        print(f"Destino: {raw_real_dir}")
        print("-" * 50)

        # Obtener archivos de imagen
        extensions = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        image_files = []

        for ext in extensions:
            image_files.extend(Path(source_dir).glob(f"*{ext}"))
            image_files.extend(Path(source_dir).glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"❌ No se encontraron imágenes en {source_dir}")
            return False

        print(f"📸 Encontradas {len(image_files)} imágenes")

        # Procesar y analizar cada imagen
        metadata_mapping = {}
        copied_count = 0

        for i, image_file in enumerate(sorted(image_files), 1):
            # Análisis automático de características
            real_metadata = self.analyze_real_image(image_file, i)

            # Generar nuevo nombre
            new_name = self.generate_real_filename(client_id, real_metadata, i)

            # Copiar archivo
            src = image_file
            dst = raw_real_dir / new_name

            shutil.copy2(str(src), str(dst))

            # Guardar metadata
            metadata_mapping[new_name] = real_metadata
            copied_count += 1

            if copied_count % 5 == 0:
                print(f"  📋 Procesadas {copied_count}/{len(image_files)} imágenes...")

        # Guardar metadata mapping
        mapping_file = metadata_dir / "real_metadata_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(metadata_mapping, f, indent=2)

        # Generar análisis de fotos reales
        real_analysis = self.analyze_real_photos_patterns(metadata_mapping)
        analysis_file = metadata_dir / "real_photos_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(real_analysis, f, indent=2)

        print(f"\n✅ IMPORTACIÓN FOTOS REALES COMPLETADA:")
        print(f"   Imágenes importadas: {copied_count}")
        print(f"   Metadata guardada: {mapping_file}")
        print(f"   Análisis guardado: {analysis_file}")

        return True

    def analyze_real_image(self, image_file, index):
        """Analiza automáticamente características de foto real"""
        filename = image_file.name.lower()

        # Detectar características del nombre del archivo
        detected_features = []

        # Análisis de iluminación
        if any(word in filename for word in ["studio", "professional", "pro"]):
            detected_features.append("studio_lighting")
        elif any(word in filename for word in ["natural", "window", "outdoor"]):
            detected_features.append("natural_lighting")
        elif any(word in filename for word in ["indoor", "room", "inside"]):
            detected_features.append("indoor_lighting")

        # Análisis de calidad
        if any(word in filename for word in ["hq", "high", "quality", "professional"]):
            detected_features.append("high_quality")
        elif any(word in filename for word in ["casual", "phone", "mobile"]):
            detected_features.append("casual_quality")

        # Análisis de sesión (agrupar por fecha/hora si es posible)
        session_group = f"session_{(index-1)//10 + 1:02d}"  # Grupos de 10

        metadata = {
            "original_filename": image_file.name,
            "import_date": datetime.now().isoformat(),
            "session_group": session_group,
            "detected_features": detected_features,
            "image_type": "real_photo",
            "purpose": "deepfacelive_compatibility",
            "file_size_mb": image_file.stat().st_size / (1024 * 1024),
        }

        return metadata

    def generate_real_filename(self, client_id, metadata, index):
        """Genera nombre para foto real"""
        session = metadata.get("session_group", f"session_{index:02d}")
        features = metadata.get("detected_features", [])
        features_str = "-".join(features[:2]) if features else "standard"

        filename = f"{client_id}_real_{index:03d}_{session}_{features_str}.png"

        return filename

    def analyze_real_photos_patterns(self, metadata_mapping):
        """Analiza patrones en fotos reales"""
        analysis = {
            "total_images": len(metadata_mapping),
            "feature_frequency": Counter(),
            "session_distribution": Counter(),
            "avg_file_size_mb": 0,
            "analysis_date": datetime.now().isoformat(),
        }

        # Analizar características
        all_features = []
        session_counts = []
        file_sizes = []

        for metadata in metadata_mapping.values():
            features = metadata.get("detected_features", [])
            all_features.extend(features)

            session = metadata.get("session_group", "unknown")
            session_counts.append(session)

            size = metadata.get("file_size_mb", 0)
            if size > 0:
                file_sizes.append(size)

        analysis["feature_frequency"] = dict(Counter(all_features))
        analysis["session_distribution"] = dict(Counter(session_counts))

        if file_sizes:
            analysis["avg_file_size_mb"] = sum(file_sizes) / len(file_sizes)

        return analysis

    def prepare_lora_dataset(self, client_id, clients_dir):
        """Prepara dataset final para LoRA con balance 85% MJ / 15% Real"""
        client_path = clients_dir / client_id
        processed_dir = client_path / "processed"
        dataset_dir = client_path / "dataset_lora"
        metadata_dir = client_path / "metadata"

        print(f"\n🎯 PREPARANDO DATASET LORA CON BALANCE PROFESIONAL")
        print("-" * 50)

        if not processed_dir.exists():
            print(
                f"❌ No hay imágenes procesadas. Ejecuta primero el procesamiento facial."
            )
            return False

        # Obtener imágenes procesadas por tipo
        processed_images = list(processed_dir.glob("*.png"))

        if not processed_images:
            print(f"❌ No hay imágenes procesadas disponibles")
            return False

        # Separar por tipo (MJ vs Real)
        mj_images = [img for img in processed_images if "_mj_" in img.name]
        real_images = [img for img in processed_images if "_real_" in img.name]

        print(f"📊 IMÁGENES DISPONIBLES:")
        print(f"   MJ: {len(mj_images)} imágenes")
        print(f"   Real: {len(real_images)} imágenes")

        # Calcular balance ideal (85% MJ / 15% Real)
        total_available = len(mj_images) + len(real_images)

        if total_available < 30:
            print(
                f"⚠️ Pocas imágenes disponibles ({total_available}). Recomendado mínimo: 50"
            )

        # Calcular distribución objetivo
        target_mj_count = int(total_available * 0.85)
        target_real_count = int(total_available * 0.15)

        # Ajustar si no hay suficientes de algún tipo
        actual_mj_count = min(len(mj_images), target_mj_count)
        actual_real_count = min(len(real_images), target_real_count)

        print(f"\n🎯 BALANCE OBJETIVO:")
        print(
            f"   MJ seleccionadas: {actual_mj_count} ({actual_mj_count/total_available*100:.1f}%)"
        )
        print(
            f"   Real seleccionadas: {actual_real_count} ({actual_real_count/total_available*100:.1f}%)"
        )

        # Seleccionar mejores imágenes (por confianza facial si está disponible)
        selected_mj = self.select_best_images(mj_images, actual_mj_count)
        selected_real = self.select_best_images(real_images, actual_real_count)

        # Crear directorio de dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copiar imágenes seleccionadas con pesos diferenciados
        dataset_metadata = {}

        # Copiar imágenes MJ (peso normal)
        for i, img_path in enumerate(selected_mj, 1):
            dst_name = f"{client_id}_mj_{i:03d}.png"
            dst_path = dataset_dir / dst_name
            shutil.copy2(img_path, dst_path)

            # Generar caption rico para MJ
            caption = self.generate_mj_caption(client_id, img_path, metadata_dir)
            caption_file = dataset_dir / f"{client_id}_mj_{i:03d}.txt"
            with open(caption_file, "w") as f:
                f.write(caption)

            dataset_metadata[dst_name] = {
                "type": "mj",
                "weight": 1.0,
                "caption": caption,
                "original_path": str(img_path),
            }

        # Copiar imágenes reales (peso reducido)
        for i, img_path in enumerate(selected_real, 1):
            dst_name = f"{client_id}_real_{i:03d}.png"
            dst_path = dataset_dir / dst_name
            shutil.copy2(img_path, dst_path)

            # Generar caption para fotos reales
            caption = self.generate_real_caption(client_id, img_path, metadata_dir)
            caption_file = dataset_dir / f"{client_id}_real_{i:03d}.txt"
            with open(caption_file, "w") as f:
                f.write(caption)

            dataset_metadata[dst_name] = {
                "type": "real",
                "weight": 0.35,  # Peso reducido para fotos reales
                "caption": caption,
                "original_path": str(img_path),
            }

        # Guardar metadata del dataset
        dataset_info = {
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "total_images": len(dataset_metadata),
            "mj_images": actual_mj_count,
            "real_images": actual_real_count,
            "balance_ratio": f"{actual_mj_count/len(dataset_metadata)*100:.1f}% MJ / {actual_real_count/len(dataset_metadata)*100:.1f}% Real",
            "weight_config": {"mj_weight": 1.0, "real_weight": 0.35},
            "recommended_training": {
                "total_steps": min(3000, len(dataset_metadata) * 25),
                "batch_size": 1,
                "learning_rate": 0.0001,
                "dataset_repeats": max(1, 500 // len(dataset_metadata)),
            },
        }

        # Guardar archivos de configuración
        dataset_config_file = dataset_dir / "dataset_config.json"
        with open(dataset_config_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        metadata_file = metadata_dir / "lora_dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        print(f"\n✅ DATASET LORA PREPARADO:")
        print(f"   Total imágenes: {len(dataset_metadata)}")
        print(f"   Balance: {dataset_info['balance_ratio']}")
        print(f"   Captions generados: {len(dataset_metadata)} archivos .txt")
        print(f"   Configuración: {dataset_config_file}")
        print(f"   Metadata: {metadata_file}")

        return True

    def select_best_images(self, image_list, target_count):
        """Selecciona las mejores imágenes basado en criterios de calidad"""
        if len(image_list) <= target_count:
            return image_list

        # Ordenar por tamaño de archivo (proxy de calidad) y tomar las mejores
        sorted_images = sorted(image_list, key=lambda x: x.stat().st_size, reverse=True)
        return sorted_images[:target_count]

    def generate_mj_caption(self, client_id, image_path, metadata_dir):
        """Genera caption rico para imagen MJ usando metadata preservada"""
        # Cargar metadata MJ
        mj_metadata_file = metadata_dir / "mj_metadata_mapping.json"

        if mj_metadata_file.exists():
            with open(mj_metadata_file, "r") as f:
                mj_metadata = json.load(f)

            # Buscar metadata para esta imagen
            for original_name, metadata in mj_metadata.items():
                if original_name in image_path.name or any(
                    part in image_path.name for part in original_name.split("_")
                ):
                    # Usar prompt y características detectadas
                    prompt_base = metadata.get(
                        "clean_prompt", f"portrait of {client_id}"
                    )
                    features = metadata.get("detected_features", [])

                    # Construir caption rico
                    caption_parts = [prompt_base]

                    if features:
                        caption_parts.append(", ".join(features))

                    caption_parts.extend(
                        ["detailed face", "high quality", "professional photography"]
                    )

                    return ", ".join(caption_parts)

        # Fallback caption
        return f"portrait of {client_id}, detailed face, high quality photography, professional headshot"

    def generate_real_caption(self, client_id, image_path, metadata_dir):
        """Genera caption para foto real enfocado en compatibilidad deepfacelive"""
        # Cargar metadata de fotos reales
        real_metadata_file = metadata_dir / "real_metadata_mapping.json"

        base_caption = f"reference photo of {client_id}, natural facial structure, geometric anchor points, detailed facial mapping"

        if real_metadata_file.exists():
            with open(real_metadata_file, "r") as f:
                real_metadata = json.load(f)

            # Buscar metadata para esta imagen
            for original_name, metadata in real_metadata.items():
                if original_name in image_path.name:
                    features = metadata.get("detected_features", [])

                    if features:
                        feature_text = ", ".join(features)
                        return f"{base_caption}, {feature_text}, compatible for face swap technology"

        return f"{base_caption}, compatible for face swap technology"

    def load_client_config(self, client_path):
        """Carga configuración del cliente"""
        config_file = client_path / "metadata" / "client_config.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_client_config(self, client_path, config):
        """Guarda configuración del cliente"""
        config_file = client_path / "metadata" / "client_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
