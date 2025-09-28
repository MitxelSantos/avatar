#!/usr/bin/env python3
"""
image_processor.py - Procesamiento facial avanzado con soporte RAW completo
Versión 3.0 - Arquitectura mejorada, bugs corregidos, escalable
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json

# Imports locales
from config import CONFIG
from utils import (
    PipelineLogger,
    ProgressTracker,
    safe_copy_file,
    safe_move_file,
    load_json_safe,
    save_json_safe,
)

# Soporte para archivos RAW
try:
    import rawpy
    import imageio

    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False


class FaceProcessor:
    """Procesador facial profesional con soporte RAW completo"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.logger = PipelineLogger("FaceProcessor", self.config.logs_dir)
        self.detector = None
        self.init_mtcnn()

        # Estadísticas de procesamiento
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "raw_conversions": 0,
            "start_time": None,
        }

    def init_mtcnn(self):
        """Inicializa detector MTCNN con manejo robusto de errores"""
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
            self.logger.info("MTCNN detector inicializado correctamente")
        except ImportError:
            self.logger.error(
                "MTCNN no disponible. Instalar: pip install mtcnn tensorflow"
            )
            self.detector = None
        except Exception as e:
            self.logger.error(f"Error inicializando MTCNN: {e}")
            self.detector = None

    def process_client_images(
        self,
        client_id: str,
        clients_dir: Path,
        source_type: str = "all",
        force_qc_params: Optional[Dict] = None,
    ) -> bool:
        """
        Procesa todas las imágenes de un cliente con arquitectura mejorada
        """
        if not self.detector:
            self.logger.error("Detector facial no disponible")
            return False

        self.stats["start_time"] = datetime.now()
        self.logger.info(f"Iniciando procesamiento para cliente: {client_id}")

        client_path = clients_dir / client_id

        # Determinar directorios fuente
        source_dirs = self._get_source_directories(client_path, source_type)
        if not source_dirs:
            self.logger.error(
                f"No se encontraron directorios fuente para {source_type}"
            )
            return False

        # Cargar metadata de conversiones RAW
        real_metadata = self._load_real_metadata(client_path)

        # Detectar si hay archivos RAW para ajustar parámetros
        has_raw_files = self._detect_raw_files(source_dirs)
        qc_params = force_qc_params or self.config.get_qc_params_for_source(
            has_raw_files
        )

        self._log_processing_config(qc_params, has_raw_files)

        # Procesar cada tipo de fuente
        all_results = []

        for img_type, source_dir in source_dirs.items():
            self.logger.info(f"Procesando {img_type.upper()}: {source_dir}")

            # Obtener archivos procesables
            image_files = self._get_processable_images(source_dir)
            if not image_files:
                self.logger.warning(f"No se encontraron imágenes en {source_dir}")
                continue

            self.logger.info(f"Encontradas {len(image_files)} imágenes para procesar")

            # Procesar con tracker de progreso
            tracker = ProgressTracker(len(image_files), f"Procesando {img_type}")

            for i, image_file in enumerate(image_files, 1):
                # Determinar archivo para procesamiento (conversión RAW si es necesario)
                processing_file = self._get_processing_file(
                    image_file, real_metadata, client_path
                )
                if not processing_file:
                    self.logger.warning(f"No se puede procesar: {image_file.name}")
                    tracker.update(status="Saltado")
                    continue

                # Procesar imagen individual
                result = self._process_single_image(
                    processing_file=processing_file,
                    original_file=image_file,
                    source_type=img_type,
                    output_dir=client_path / "processed",
                    qc_params=qc_params,
                    index=i,
                )

                if result:
                    all_results.append(result)
                    if result["keep"]:
                        self.stats["successful"] += 1
                        tracker.update(
                            status=f"Aceptada (conf: {result['face_confidence']:.2f})"
                        )
                    else:
                        self.stats["failed"] += 1
                        issues = ", ".join(result["qc_issues"])
                        tracker.update(status=f"Rechazada ({issues})")
                else:
                    self.stats["failed"] += 1
                    tracker.update(status="Error")

                self.stats["total_processed"] += 1

            tracker.finish(f"Completado {img_type}")

        # Generar reporte y finalizar
        if all_results:
            success = self._finalize_processing(all_results, client_path, client_id)
            self._log_final_stats()
            return success
        else:
            self.logger.error("No se procesaron imágenes exitosamente")
            return False

    def _get_source_directories(
        self, client_path: Path, source_type: str
    ) -> Dict[str, Path]:
        """Obtiene directorios fuente según el tipo especificado"""
        source_dirs = {}

        if source_type in ["all", "mj"] and (client_path / "raw_mj").exists():
            source_dirs["mj"] = client_path / "raw_mj"

        if source_type in ["all", "real"] and (client_path / "raw_real").exists():
            source_dirs["real"] = client_path / "raw_real"

        return source_dirs

    def _load_real_metadata(self, client_path: Path) -> Dict:
        """Carga metadata de fotos reales para conversiones RAW"""
        metadata_file = client_path / "metadata" / "real_metadata_mapping.json"
        return load_json_safe(metadata_file, {}, self.logger)

    def _detect_raw_files(self, source_dirs: Dict[str, Path]) -> bool:
        """Detecta si hay archivos RAW en los directorios fuente"""
        raw_extensions = [
            ext.lower() for ext in self.config.supported_extensions["raw"]
        ]

        for source_dir in source_dirs.values():
            for file_path in source_dir.iterdir():
                if file_path.suffix.lower() in raw_extensions:
                    return True

        return False

    def _log_processing_config(self, qc_params: Dict, has_raw_files: bool):
        """Registra configuración de procesamiento"""
        self.logger.info("Configuración de procesamiento:")
        for key, value in qc_params.items():
            self.logger.info(f"  {key}: {value}")

        if has_raw_files:
            self.logger.info("Usando parámetros optimizados para archivos RAW")

        if not RAW_SUPPORT and has_raw_files:
            self.logger.warning("Archivos RAW detectados pero rawpy no está disponible")

    def _get_processable_images(self, source_dir: Path) -> List[Path]:
        """Obtiene lista de imágenes procesables evitando duplicados"""
        all_extensions = (
            self.config.supported_extensions["standard"]
            + self.config.supported_extensions["raw"]
        )

        image_files = []
        seen_files = set()

        for file_path in source_dir.iterdir():
            if file_path.is_file():
                # Verificar extensión (case-insensitive)
                if file_path.suffix.lower() in [ext.lower() for ext in all_extensions]:
                    # Usar path absoluto para evitar duplicados
                    abs_path = file_path.resolve()
                    if abs_path not in seen_files:
                        seen_files.add(abs_path)
                        image_files.append(file_path)

        return sorted(image_files)

    def _get_processing_file(
        self, original_file: Path, real_metadata: Dict, client_path: Path
    ) -> Optional[Path]:
        """
        Determina qué archivo usar para procesamiento facial
        Maneja conversión RAW transparentemente
        """
        # Verificar si es archivo RAW
        is_raw = original_file.suffix.lower() in [
            ext.lower() for ext in self.config.supported_extensions["raw"]
        ]

        if not is_raw:
            # Archivo estándar - usar directamente
            return original_file

        # Archivo RAW - buscar conversión temporal en metadata
        filename = original_file.name
        if filename in real_metadata:
            metadata = real_metadata[filename]
            temp_path = metadata.get("temp_conversion_path")

            if temp_path and Path(temp_path).exists():
                self.logger.debug(
                    f"Usando conversión temporal para RAW: {Path(temp_path).name}"
                )
                return Path(temp_path)

        # Si no hay conversión temporal, convertir on-the-fly
        if RAW_SUPPORT:
            self.logger.info(f"Convirtiendo RAW on-the-fly: {original_file.name}")
            temp_dir = client_path / "temp_processing"
            temp_file = self._convert_raw_on_demand(original_file, temp_dir)
            if temp_file:
                self.stats["raw_conversions"] += 1
            return temp_file
        else:
            self.logger.error(
                f"Archivo RAW sin conversión disponible y rawpy no instalado"
            )
            return None

    def _convert_raw_on_demand(self, raw_path: Path, temp_dir: Path) -> Optional[Path]:
        """
        Convierte archivo RAW temporalmente para procesamiento
        VERSIÓN CORREGIDA - Compatible con rawpy estándar
        """
        if not RAW_SUPPORT:
            self.logger.error("rawpy no está disponible")
            return None

        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"processing_{raw_path.stem}.jpg"
        temp_path = temp_dir / temp_name

        try:
            with rawpy.imread(str(raw_path)) as raw:
                # Configuración compatible - solo parámetros estándar
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8,
                    bright=1.0,
                )

            imageio.imwrite(str(temp_path), rgb, quality=95)
            self.logger.debug(f"Conversión RAW exitosa: {temp_path}")
            return temp_path

        except Exception as e:
            self.logger.warning(f"Error convirtiendo RAW {raw_path.name}: {str(e)}")
            self.logger.info("Intentando configuración básica...")

            # Configuración de respaldo
            try:
                with rawpy.imread(str(raw_path)) as raw:
                    rgb = raw.postprocess()  # Usar configuración por defecto

                imageio.imwrite(str(temp_path), rgb, quality=90)
                self.logger.info(
                    f"Conversión RAW exitosa con configuración básica: {temp_path}"
                )
                return temp_path

            except Exception as e2:
                self.logger.error(f"Conversión RAW falló completamente: {str(e2)}")
                return None

    def _process_single_image(
        self,
        processing_file: Path,
        original_file: Path,
        source_type: str,
        output_dir: Path,
        qc_params: Dict,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Procesa una sola imagen con detección facial completa
        VERSIÓN CORREGIDA - Estructura de datos consistente
        """
        try:
            # Estructura de resultado consistente
            result_template = {
                "original_path": str(original_file),
                "processing_path": str(processing_file),
                "output_path": "",
                "filename": "",
                "source_type": source_type,
                "face_confidence": 0.0,
                "face_box": [],
                "face_keypoints": {},
                "qc_issues": [],
                "keep": False,
                "processed_date": datetime.now().isoformat(),
                "file_size_kb": 0.0,
                "image_dimensions": "0x0",
                "is_raw_source": str(original_file) != str(processing_file),
            }

            # Cargar imagen
            img = cv2.imread(str(processing_file))
            if img is None:
                result_template["qc_issues"] = ["Could not load image"]
                return result_template

            # Convertir a RGB para MTCNN
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detectar rostros
            results = self.detector.detect_faces(rgb_img)

            if not results:
                result_template["qc_issues"] = ["No face detected"]
                return result_template

            # Obtener mejor rostro (mayor confianza)
            face = max(results, key=lambda x: x["confidence"])
            confidence = face["confidence"]

            # Verificar confianza mínima
            if confidence < qc_params["face_confidence_threshold"]:
                result_template.update(
                    {
                        "face_confidence": confidence,
                        "qc_issues": ["Low face confidence"],
                    }
                )
                return result_template

            # Recortar rostro en formato cuadrado
            cropped = self._crop_face_square(rgb_img, face, qc_params)
            if cropped is None:
                result_template.update(
                    {"face_confidence": confidence, "qc_issues": ["Crop failed"]}
                )
                return result_template

            # Redimensionar y mejorar
            final_img = self._resize_and_enhance(cropped)

            # Generar nombre de archivo de salida
            output_filename = self._generate_output_filename(
                original_file, source_type, confidence, index
            )

            # Crear directorio de salida y guardar
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename

            final_img.save(str(output_path), "PNG", quality=95, optimize=True)

            # Control de calidad
            qc_issues = self._quality_check(output_path, final_img, qc_params)

            # Resultado completo
            result_template.update(
                {
                    "output_path": str(output_path),
                    "filename": str(output_filename),
                    "face_confidence": float(confidence),
                    "face_box": face["box"],
                    "face_keypoints": face.get("keypoints", {}),
                    "qc_issues": qc_issues,
                    "keep": len(qc_issues) == 0,
                    "file_size_kb": float(output_path.stat().st_size / 1024),
                    "image_dimensions": f"{final_img.size[0]}x{final_img.size[1]}",
                }
            )

            return result_template

        except Exception as e:
            self.logger.error(f"Error procesando {original_file.name}: {str(e)}")
            result_template["qc_issues"] = [f"Processing error: {str(e)}"]
            return result_template

    def _crop_face_square(
        self, img: np.ndarray, face_info: Dict, qc_params: Dict
    ) -> Optional[np.ndarray]:
        """Recorta rostro en formato cuadrado con padding optimizado"""
        x, y, w, h = face_info["box"]

        # Calcular centro del rostro
        center_x = x + w // 2
        center_y = y + h // 2

        # Determinar tamaño de recorte con padding
        face_size = max(w, h)
        crop_size = int(face_size * qc_params["face_padding_factor"])

        # Asegurar que el recorte no exceda los límites de la imagen
        img_h, img_w = img.shape[:2]

        half_crop = crop_size // 2
        left = max(0, center_x - half_crop)
        top = max(0, center_y - half_crop)
        right = min(img_w, center_x + half_crop)
        bottom = min(img_h, center_y + half_crop)

        # Ajustar para mantener formato cuadrado
        crop_w = right - left
        crop_h = bottom - top

        if crop_w != crop_h:
            if crop_w < crop_h:
                diff = crop_h - crop_w
                left = max(0, left - diff // 2)
                right = min(img_w, right + diff // 2)
            else:
                diff = crop_w - crop_h
                top = max(0, top - diff // 2)
                bottom = min(img_h, bottom + diff // 2)

        # Recorte final
        cropped = img[top:bottom, left:right]

        if cropped.size == 0:
            self.logger.warning("Recorte vacío generado")
            return None

        return cropped

    def _resize_and_enhance(self, img_array: np.ndarray) -> Image.Image:
        """Redimensiona a 1024x1024 y aplica mejoras sutiles"""
        # Convertir a PIL
        pil_img = Image.fromarray(img_array)

        # Redimensionar con alta calidad
        pil_img = pil_img.resize((1024, 1024), Image.Resampling.LANCZOS)

        # Mejoras sutiles
        # Ligero sharpening
        pil_img = pil_img.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=3)
        )

        # Ligero aumento de contraste
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.05)

        return pil_img

    def _quality_check(
        self, image_path: Path, img: Image.Image, qc_params: Dict
    ) -> List[str]:
        """Control de calidad automatizado con logging detallado"""
        issues = []

        # Verificar tamaño
        if img.size != (1024, 1024):
            issues.append(f"Wrong size: {img.size}")

        # Verificar tamaño de archivo
        if image_path.exists():
            file_size_kb = image_path.stat().st_size / 1024
            if file_size_kb < qc_params["min_file_size_kb"]:
                issues.append(f"File too small: {file_size_kb:.1f}KB")
            elif file_size_kb > qc_params["max_file_size_mb"] * 1024:
                issues.append(f"File too large: {file_size_kb/1024:.1f}MB")

        # Convertir a escala de grises para análisis
        gray_array = np.array(img.convert("L"))

        # Verificar brillo
        avg_brightness = np.mean(gray_array)
        if avg_brightness < qc_params["min_brightness"]:
            issues.append("Too dark")
        elif avg_brightness > qc_params["max_brightness"]:
            issues.append("Too bright")

        # Verificar contraste
        contrast = np.std(gray_array)
        if contrast < qc_params["min_contrast"]:
            issues.append("Low contrast")

        # Verificar desenfoque (Laplacian variance)
        blur_score = cv2.Laplacian(gray_array, cv2.CV_64F).var()
        if blur_score < qc_params["blur_threshold"]:
            issues.append(f"Blurry (score: {blur_score:.1f})")

        return issues

    def _generate_output_filename(
        self, original_path: Path, source_type: str, confidence: float, index: int
    ) -> str:
        """Genera nombre de archivo de salida descriptivo"""
        original_name = original_path.stem

        # Incluir información del tipo de fuente
        if source_type == "mj":
            # Mantener información útil del nombre MJ original
            parts = original_name.split("_")
            if len(parts) > 3:
                descriptor = "_".join(parts[1:4])  # Tomar partes del prompt
            else:
                descriptor = original_name
        else:
            # Para archivos reales, incluir info de formato
            original_ext = original_path.suffix.upper()
            format_info = (
                "RAW"
                if original_ext in [".NEF", ".CR2", ".ARW", ".DNG", ".RAF", ".ORF"]
                else "STD"
            )
            descriptor = f"real_{index:03d}_{format_info}"

        # Nombre final con confianza
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"{source_type}_{descriptor}_conf{confidence:.2f}_{timestamp}.png"

        return filename

    def _finalize_processing(
        self, results: List[Dict], client_path: Path, client_id: str
    ) -> bool:
        """Finaliza el procesamiento con reporte y limpieza"""
        try:
            # Convertir a DataFrame de forma segura
            df = pd.DataFrame(results)

            # Generar reporte
            report = self._generate_processing_report(df, client_id)

            # Mover imágenes rechazadas
            self._move_rejected_images(df, client_path)

            # Guardar metadata completa
            self._save_processing_metadata(df, report, client_path)

            # Limpiar archivos temporales
            self._cleanup_temp_files(client_path)

            # Mostrar resultados finales
            self._show_final_results(report)

            return True

        except Exception as e:
            self.logger.error(f"Error finalizando procesamiento: {e}")
            return False

    def _generate_processing_report(
        self, df: pd.DataFrame, client_id: str
    ) -> Dict[str, Any]:
        """Genera reporte completo del procesamiento"""
        total_processed = len(df)
        successful = len(df[df["keep"] == True])
        rejected = total_processed - successful

        # Breakdown por tipo de fuente
        by_source_type = {}
        if "source_type" in df.columns:
            by_source_type = (
                df[df["keep"] == True]["source_type"].value_counts().to_dict()
            )

        # Conteo de archivos RAW procesados
        raw_processed = (
            len(df[df.get("is_raw_source", False) == True])
            if "is_raw_source" in df.columns
            else 0
        )

        # Análisis de problemas de QC
        all_issues = []
        for issues_list in df["qc_issues"]:
            if isinstance(issues_list, list):
                all_issues.extend(issues_list)

        from collections import Counter

        issues_summary = Counter(all_issues)

        # Estadísticas de confianza facial
        confidence_stats = {
            "mean": df["face_confidence"].mean(),
            "min": df["face_confidence"].min(),
            "max": df["face_confidence"].max(),
            "std": df["face_confidence"].std(),
        }

        report = {
            "client_id": client_id,
            "processing_date": datetime.now().isoformat(),
            "processing_summary": {
                "total_processed": total_processed,
                "successful": successful,
                "rejected": rejected,
                "success_rate": (
                    f"{(successful/total_processed)*100:.1f}%"
                    if total_processed > 0
                    else "0%"
                ),
                "raw_files_processed": raw_processed,
                "raw_conversions": self.stats["raw_conversions"],
            },
            "by_source_type": by_source_type,
            "qc_issues_summary": dict(issues_summary),
            "face_confidence_stats": confidence_stats,
            "ready_for_lora": successful >= 30,  # Reducido threshold
            "lora_training_recommendations": {
                "total_images": successful,
                "recommended_steps": min(3500, successful * 30),
                "batch_size": 1,
                "learning_rate": 0.0001 if successful < 100 else 0.00008,
            },
        }

        return report

    def _move_rejected_images(self, df: pd.DataFrame, client_path: Path):
        """Mueve imágenes rechazadas a carpeta separada - VERSIÓN CORREGIDA"""
        rejected_df = df[df["keep"] == False]

        if len(rejected_df) == 0:
            self.logger.info("No hay imágenes rechazadas que mover")
            return

        rejected_dir = client_path / "rejected"
        rejected_dir.mkdir(exist_ok=True)

        moved_count = 0

        for _, row in rejected_df.iterrows():
            try:
                output_path = row.get("output_path")
                filename = row.get("filename")

                # Verificar que los datos sean válidos
                if not output_path or not filename:
                    continue

                # Verificar que sean strings válidas
                if not isinstance(output_path, str) or not isinstance(filename, str):
                    continue

                src = Path(output_path)
                dst = rejected_dir / filename

                # Verificar que el archivo fuente existe
                if not src.exists():
                    continue

                # Manejar duplicados en destino
                if dst.exists():
                    timestamp = datetime.now().strftime("%H%M%S")
                    dst = rejected_dir / f"{dst.stem}_dup{timestamp}{dst.suffix}"

                # Mover archivo
                if safe_move_file(src, dst, self.logger):
                    moved_count += 1

            except Exception as e:
                self.logger.warning(f"Error moviendo imagen rechazada: {e}")
                continue

        self.logger.info(
            f"Movidas {moved_count}/{len(rejected_df)} imágenes rechazadas"
        )

    def _save_processing_metadata(
        self, df: pd.DataFrame, report: Dict, client_path: Path
    ):
        """Guarda metadata completa del procesamiento"""
        metadata_dir = client_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # Guardar CSV con toda la información
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = metadata_dir / f"processing_log_{timestamp}.csv"

        try:
            df.to_csv(csv_file, index=False, encoding="utf-8")
            self.logger.info(f"CSV guardado: {csv_file}")
        except Exception as e:
            self.logger.error(f"Error guardando CSV: {e}")

        # Guardar reporte JSON
        report_file = metadata_dir / f"processing_report_{timestamp}.json"
        if save_json_safe(report, report_file, self.logger):
            self.logger.info(f"Reporte guardado: {report_file}")

        # Actualizar resumen general
        summary_file = metadata_dir / "processing_summary.json"
        save_json_safe(report, summary_file, self.logger)

    def _cleanup_temp_files(self, client_path: Path):
        """Limpia archivos temporales generados durante el procesamiento"""
        temp_dirs = [
            client_path / "temp_processing",
            client_path / "temp_raw_conversion",
        ]

        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Archivos temporales eliminados: {temp_dir}")
                except Exception as e:
                    self.logger.warning(
                        f"No se pudieron eliminar archivos temporales: {e}"
                    )

    def _show_final_results(self, report: Dict):
        """Muestra resultados finales del procesamiento"""
        print(f"\n" + "=" * 60)
        print(f"🎉 PROCESAMIENTO FACIAL COMPLETADO")
        print(f"=" * 60)
        print(f"📊 Total procesadas: {report['processing_summary']['total_processed']}")
        print(f"✅ Exitosas: {report['processing_summary']['successful']}")
        print(f"❌ Rechazadas: {report['processing_summary']['rejected']}")
        print(f"📈 Tasa de éxito: {report['processing_summary']['success_rate']}")
        print(f"🎯 Listo para LoRA: {'SÍ' if report['ready_for_lora'] else 'NO'}")

        # Estadísticas RAW
        raw_processed = report["processing_summary"].get("raw_conversions", 0)
        if raw_processed > 0:
            print(f"📸 Conversiones RAW: {raw_processed}")

        # Mostrar problemas más comunes
        if report["qc_issues_summary"]:
            print(f"\n🔍 PROBLEMAS MÁS COMUNES:")
            for issue, count in list(report["qc_issues_summary"].items())[:5]:
                print(f"   {issue}: {count} imágenes")

        # Breakdown por tipo
        if report["by_source_type"]:
            print(f"\n📋 DISTRIBUCIÓN POR TIPO:")
            for img_type, count in report["by_source_type"].items():
                percentage = (count / report["processing_summary"]["successful"]) * 100
                print(f"   {img_type.upper()}: {count} imágenes ({percentage:.1f}%)")

        # Estadísticas de confianza facial
        conf_stats = report["face_confidence_stats"]
        print(f"\n📊 CONFIANZA FACIAL:")
        print(f"   Promedio: {conf_stats['mean']:.2f}")
        print(f"   Rango: {conf_stats['min']:.2f} - {conf_stats['max']:.2f}")

        print(f"\n💡 Las imágenes procesadas están listas para el siguiente paso")

    def _log_final_stats(self):
        """Registra estadísticas finales en el log"""
        if self.stats["start_time"]:
            duration = datetime.now() - self.stats["start_time"]
            self.logger.info(f"Procesamiento completado en {duration}")

        self.logger.info(f"Estadísticas finales:")
        self.logger.info(f"  Total procesadas: {self.stats['total_processed']}")
        self.logger.info(f"  Exitosas: {self.stats['successful']}")
        self.logger.info(f"  Fallidas: {self.stats['failed']}")
        self.logger.info(f"  Conversiones RAW: {self.stats['raw_conversions']}")
