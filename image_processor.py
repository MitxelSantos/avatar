#!/usr/bin/env python3
"""
image_processor.py
Módulo especializado para procesamiento facial con MTCNN
Detección, recorte, redimensionamiento y control de calidad
INCLUYE SOPORTE AUTOMÁTICO PARA ARCHIVOS RAW (.NEF, .CR2, etc.)
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from datetime import datetime
import json


class FaceProcessor:
    def __init__(self):
        self.detector = None
        self.init_mtcnn()

    def init_mtcnn(self):
        """Inicializa detector MTCNN"""
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
            print("✅ MTCNN detector loaded successfully")
        except ImportError:
            print("❌ MTCNN not available. Install: pip install mtcnn tensorflow")
            self.detector = None

    def load_image_with_raw_support(self, image_path):
        """Carga imagen con soporte automático para RAW (.NEF, .CR2, etc.)"""
        path_str = str(image_path)
        file_ext = Path(path_str).suffix.lower()

        # Extensiones RAW que necesitan rawpy (case-insensitive)
        raw_extensions = {
            ".NEF",
            ".cr2",
            ".cr3",
            ".arw",
            ".dng",
            ".orf",
            ".raf",
            ".rw2",
        }

        if file_ext in raw_extensions:
            print(f"      🔍 Archivo RAW detectado: {file_ext}")
            return self.convert_raw_to_bgr(path_str)
        else:
            # Formato estándar, usar OpenCV
            return cv2.imread(path_str)

    def convert_raw_to_bgr(self, raw_path):
        """Convierte archivo RAW a BGR array (formato OpenCV)"""
        try:
            import rawpy

            print(f"      🔄 Procesando RAW...")

            with rawpy.imread(raw_path) as raw:
                # Conversión optimizada para calidad facial
                rgb = raw.postprocess(
                    use_camera_wb=True,  # Usar balance de blancos de cámara
                    half_size=False,  # Resolución completa
                    no_auto_bright=True,  # Sin auto-exposición
                    output_bps=8,  # 8 bits por canal
                    gamma=(1, 1),  # Sin corrección gamma artificial
                    bright=1.0,  # Brillo natural
                    highlight_mode=1,  # Recuperar highlights
                    use_auto_wb=False,  # No auto white balance
                    output_color=rawpy.ColorSpace.sRGB,  # Espacio color estándar
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # Mejor calidad
                )

            # Convertir RGB a BGR (formato que espera OpenCV)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            print(
                f"      ✅ RAW convertido exitosamente: {rgb.shape[1]}x{rgb.shape[0]}"
            )
            return bgr

        except ImportError:
            print(f"      ❌ rawpy no instalado. Ejecuta: pip install rawpy")
            print(f"      💡 Sin rawpy, no se pueden procesar archivos RAW")
            return None
        except Exception as e:
            print(f"      ❌ Error procesando RAW {Path(raw_path).name}: {e}")
            return None

    def process_client_images(
        self, client_id, clients_dir, source_type="all", qc_params=None
    ):
        """Procesa todas las imágenes de un cliente con detección facial completa"""
        client_path = clients_dir / client_id

        if not self.detector:
            print("❌ Detector facial no disponible")
            return False

        # Directorios fuente según tipo
        source_dirs = {}

        if source_type in ["all", "mj"] and (client_path / "raw_mj").exists():
            source_dirs["mj"] = client_path / "raw_mj"

        if source_type in ["all", "real"] and (client_path / "raw_real").exists():
            source_dirs["real"] = client_path / "raw_real"

        if not source_dirs:
            print(f"❌ No se encontraron directorios fuente para {source_type}")
            return False

        # Mostrar parámetros de calidad
        print(f"⚙️ PARÁMETROS DE CALIDAD:")
        for key, value in qc_params.items():
            readable_key = key.replace("_", " ").title()
            print(f"   {readable_key}: {value}")
        print()

        # Procesar cada tipo de fuente
        all_results = []
        successful_count = 0
        failed_count = 0

        for img_type, source_dir in source_dirs.items():
            print(f"🔄 PROCESANDO {img_type.upper()}")
            print("-" * 30)

            # Obtener archivos de imagen (incluyendo RAW) - SIN DUPLICADOS
            extensions = {
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff",
                ".tif",  # Estándar
                ".NEF",  # Nikon RAW
                ".cr2",
                ".cr3",  # Canon RAW
                ".arw",  # Sony RAW
                ".dng",  # Adobe DNG
                ".orf",  # Olympus RAW
                ".raf",  # Fujifilm RAW
                ".rw2",  # Panasonic RAW
                ".pef",  # Pentax RAW
                ".srw",  # Samsung RAW
                ".3fr",
                ".fff",
                ".dcr",
                ".kdc",
                ".srf",
                ".sr2",  # Otros RAW
            }

            # Buscar archivos con extensiones case-insensitive (sin duplicados)
            image_files = [
                f
                for f in source_dir.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]

            if not image_files:
                print(f"❌ No se encontraron imágenes en {source_dir}")
                continue

            print(f"📸 Encontradas {len(image_files)} imágenes")

            # Procesar cada imagen
            for i, image_file in enumerate(image_files, 1):
                print(f"\n📷 [{i:3d}/{len(image_files)}] {image_file.name}")

                result = self.process_single_image(
                    image_path=str(image_file),
                    source_type=img_type,
                    output_dir=client_path / "processed",
                    qc_params=qc_params,
                    index=i,
                )

                if result:
                    all_results.append(result)
                    if result["keep"]:
                        successful_count += 1
                        print(
                            f"   ✅ ACEPTADA - Confianza: {result['face_confidence']:.2f}"
                        )
                    else:
                        failed_count += 1
                        issues = ", ".join(result["qc_issues"])
                        print(f"   ❌ RECHAZADA - Motivos: {issues}")
                else:
                    failed_count += 1
                    print(f"   ❌ FALLÓ - No se pudo procesar")

                # Mostrar progreso cada 10 imágenes
                if i % 10 == 0:
                    print(
                        f"\n📊 PROGRESO: {i}/{len(image_files)} procesadas | ✅{successful_count} ❌{failed_count}"
                    )

        # Generar reporte y mover archivos
        if all_results:
            df = pd.DataFrame(all_results)

            # Generar reporte
            report = self.generate_processing_report(df, client_id)

            # Mover imágenes rechazadas
            self.move_rejected_images(df, client_path)

            # Preparar dataset para LoRA
            self.prepare_processed_dataset(df, client_path)

            # Guardar metadata completa
            self.save_processing_metadata(df, report, client_path)

            # Mostrar resultados finales
            self.show_final_results(report)

            return True
        else:
            print("❌ No se procesaron imágenes exitosamente")
            return False

    def process_single_image(
        self, image_path, source_type, output_dir, qc_params, index=1
    ):
        """Procesa una sola imagen con detección facial completa"""
        try:
            print(f"      🔍 Detectando rostro...")

            # Cargar imagen (soporte RAW automático)
            img = self.load_image_with_raw_support(image_path)
            if img is None:
                print(f"      ❌ No se pudo cargar la imagen")
                return None

            # Convertir a RGB para MTCNN
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detectar rostros
            results = self.detector.detect_faces(rgb_img)

            if not results:
                print(f"      ❌ No se detectó rostro")
                return None

            # Obtener mejor rostro (mayor confianza)
            face = max(results, key=lambda x: x["confidence"])
            confidence = face["confidence"]

            print(f"      ✅ Rostro detectado - Confianza: {confidence:.2f}")

            # Verificar confianza mínima
            if confidence < qc_params["face_confidence_threshold"]:
                print(
                    f"      ❌ Confianza muy baja (mín: {qc_params['face_confidence_threshold']})"
                )
                return {
                    "original_path": image_path,
                    "filename": os.path.basename(image_path),
                    "source_type": source_type,
                    "face_confidence": confidence,
                    "qc_issues": ["Low face confidence"],
                    "keep": False,
                    "processed_date": datetime.now().isoformat(),
                }

            print(f"      🔄 Recortando rostro...")

            # Recortar rostro en formato cuadrado
            cropped = self.crop_face_square(rgb_img, face, qc_params)
            if cropped is None:
                print(f"      ❌ Error en recorte")
                return None

            print(f"      🎨 Redimensionando a 1024x1024...")

            # Redimensionar y mejorar
            final_img = self.resize_and_enhance(cropped)

            # Generar nombre de archivo de salida
            output_filename = self.generate_output_filename(
                image_path, source_type, confidence, index
            )

            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_filename

            print(f"      💾 Guardando como: {output_filename}")

            # Guardar imagen
            final_img.save(str(output_path), "PNG", quality=95, optimize=True)

            print(f"      🔍 Control de calidad...")

            # Control de calidad
            qc_issues = self.quality_check(str(output_path), final_img, qc_params)

            # Metadata completa
            metadata = {
                "original_path": image_path,
                "output_path": str(output_path),
                "filename": output_filename,
                "source_type": source_type,
                "face_confidence": confidence,
                "face_box": face["box"],
                "face_keypoints": face.get("keypoints", {}),
                "qc_issues": qc_issues,
                "keep": len(qc_issues) == 0,
                "processed_date": datetime.now().isoformat(),
                "file_size_kb": os.path.getsize(output_path) / 1024,
                "image_dimensions": f"{final_img.size[0]}x{final_img.size[1]}",
                "is_raw_source": Path(image_path).suffix.lower()
                in {".NEF", ".cr2", ".cr3", ".arw", ".dng", ".orf", ".raf", ".rw2"},
            }

            return metadata

        except Exception as e:
            print(f"      ❌ Error procesando: {str(e)}")
            return None

    def crop_face_square(self, img, face_info, qc_params):
        """Recorta rostro en formato cuadrado con padding"""
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
            print(f"      ❌ Recorte vacío")
            return None

        print(f"      ✅ Recortado: {cropped.shape[1]}x{cropped.shape[0]}")
        return cropped

    def resize_and_enhance(self, img_array):
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

    def quality_check(self, image_path, img, qc_params):
        """Control de calidad automatizado con logging detallado"""
        issues = []

        # Verificar tamaño
        if img.size != (1024, 1024):
            issues.append(f"Wrong size: {img.size}")
            print(f"      ❌ Tamaño incorrecto: {img.size}")

        # Verificar tamaño de archivo
        if os.path.exists(image_path):
            file_size_kb = os.path.getsize(image_path) / 1024
            if file_size_kb < qc_params["min_file_size_kb"]:
                issues.append(f"File too small: {file_size_kb:.1f}KB")
                print(f"      ❌ Archivo muy pequeño: {file_size_kb:.1f}KB")
            elif file_size_kb > qc_params["max_file_size_mb"] * 1024:
                issues.append(f"File too large: {file_size_kb/1024:.1f}MB")
                print(f"      ❌ Archivo muy grande: {file_size_kb/1024:.1f}MB")

        # Convertir a escala de grises para análisis
        gray_array = np.array(img.convert("L"))

        # Verificar brillo
        avg_brightness = np.mean(gray_array)
        if avg_brightness < qc_params["min_brightness"]:
            issues.append("Too dark")
            print(f"      ❌ Muy oscuro: {avg_brightness:.1f}")
        elif avg_brightness > qc_params["max_brightness"]:
            issues.append("Too bright")
            print(f"      ❌ Muy brillante: {avg_brightness:.1f}")
        else:
            print(f"      ✅ Brillo OK: {avg_brightness:.1f}")

        # Verificar contraste
        contrast = np.std(gray_array)
        if contrast < qc_params["min_contrast"]:
            issues.append("Low contrast")
            print(f"      ❌ Contraste bajo: {contrast:.1f}")
        else:
            print(f"      ✅ Contraste OK: {contrast:.1f}")

        # Verificar desenfoque (Laplacian variance)
        blur_score = cv2.Laplacian(gray_array, cv2.CV_64F).var()
        if blur_score < qc_params["blur_threshold"]:
            issues.append(f"Blurry (score: {blur_score:.1f})")
            print(f"      ❌ Desenfocado: {blur_score:.1f}")
        else:
            print(f"      ✅ Nitidez OK: {blur_score:.1f}")

        if not issues:
            print(f"      ✅ Pasa todos los controles de calidad")

        return issues

    def generate_output_filename(self, original_path, source_type, confidence, index):
        """Genera nombre de archivo de salida"""
        original_name = Path(original_path).stem

        # Extraer características del nombre original si es MJ
        if source_type == "mj":
            # Mantener información útil del nombre MJ original
            parts = original_name.split("_")
            if len(parts) > 3:
                descriptor = "_".join(parts[1:4])  # Tomar partes del prompt
            else:
                descriptor = original_name
        else:
            descriptor = f"real_{index:03d}"

        # Indicar si viene de RAW (case-insensitive)
        is_raw = Path(original_path).suffix.lower() in {
            ".NEF",
            ".cr2",
            ".cr3",
            ".arw",
            ".dng",
            ".orf",
            ".raf",
            ".rw2",
        }
        raw_suffix = "_RAW" if is_raw else ""

        # Nombre final con confianza
        filename = f"{source_type}_{descriptor}{raw_suffix}_conf{confidence:.2f}.png"

        return filename

    def generate_processing_report(self, df, client_id):
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

        # Análisis de problemas de QC
        all_issues = []
        for issues_list in df["qc_issues"]:
            if issues_list:
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

        # Estadísticas RAW vs estándar
        raw_stats = {}
        if "is_raw_source" in df.columns:
            raw_count = len(df[df["is_raw_source"] == True])
            standard_count = total_processed - raw_count
            raw_stats = {
                "raw_files_processed": raw_count,
                "standard_files_processed": standard_count,
                "raw_success_rate": len(
                    df[(df["is_raw_source"] == True) & (df["keep"] == True)]
                )
                / max(1, raw_count),
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
            },
            "by_source_type": by_source_type,
            "raw_processing_stats": raw_stats,
            "qc_issues_summary": dict(issues_summary),
            "face_confidence_stats": confidence_stats,
            "ready_for_lora": successful >= 50,
            "lora_training_recommendations": {
                "total_images": successful,
                "recommended_steps": min(3000, successful * 25),
                "batch_size": 1 if successful < 100 else 2,
                "learning_rate": 0.0001,
            },
        }

        return report

    def move_rejected_images(self, df, client_path):
        """Mueve imágenes rechazadas a carpeta separada"""
        rejected_df = df[df["keep"] == False]

        if len(rejected_df) == 0:
            print(f"\n✅ No hay imágenes rechazadas que mover")
            return

        print(f"\n📦 MOVIENDO {len(rejected_df)} IMÁGENES RECHAZADAS...")

        rejected_dir = client_path / "rejected"
        rejected_dir.mkdir(exist_ok=True)

        for i, (_, row) in enumerate(rejected_df.iterrows(), 1):
            src = Path(row["output_path"])
            dst = rejected_dir / row["filename"]

            if src.exists():
                try:
                    if dst.exists():
                        # Manejar duplicados
                        timestamp = datetime.now().strftime("%H%M%S")
                        dst = rejected_dir / f"{dst.stem}_dup{timestamp}{dst.suffix}"

                    src.rename(dst)
                    print(f"  [{i:3d}] ✅ Movida: {row['filename']}")
                except Exception as e:
                    print(f"  [{i:3d}] ❌ Error moviendo {row['filename']}: {str(e)}")

    def prepare_processed_dataset(self, df, client_path):
        """Prepara dataset de imágenes procesadas exitosamente"""
        accepted_df = df[df["keep"] == True].copy()

        if len(accepted_df) == 0:
            print(f"\n❌ No hay imágenes aceptadas para el dataset")
            return

        print(f"\n🎯 PREPARANDO DATASET PROCESADO CON {len(accepted_df)} IMÁGENES...")

        processed_dir = client_path / "processed"

        # Las imágenes ya están en processed/, solo necesitamos organizar metadata
        print(f"✅ {len(accepted_df)} imágenes listas en {processed_dir}")

        # Estadísticas por tipo
        if "source_type" in accepted_df.columns:
            type_counts = accepted_df["source_type"].value_counts()
            for img_type, count in type_counts.items():
                percentage = (count / len(accepted_df)) * 100
                print(f"   {img_type.upper()}: {count} imágenes ({percentage:.1f}%)")

        # Estadísticas RAW
        if "is_raw_source" in accepted_df.columns:
            raw_count = len(accepted_df[accepted_df["is_raw_source"] == True])
            if raw_count > 0:
                raw_percentage = (raw_count / len(accepted_df)) * 100
                print(
                    f"   RAW: {raw_count} imágenes ({raw_percentage:.1f}%) - Máxima calidad preservada"
                )

    def save_processing_metadata(self, df, report, client_path):
        """Guarda metadata completa del procesamiento"""
        metadata_dir = client_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # Guardar CSV con toda la información
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = metadata_dir / f"processing_log_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # Guardar reporte JSON
        report_file = metadata_dir / f"processing_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Actualizar resumen general
        summary_file = metadata_dir / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📋 METADATA GUARDADA:")
        print(f"   CSV: {csv_file}")
        print(f"   Reporte: {report_file}")
        print(f"   Resumen: {summary_file}")

    def show_final_results(self, report):
        """Muestra resultados finales del procesamiento"""
        print(f"\n" + "=" * 60)
        print(f"🎉 PROCESAMIENTO FACIAL COMPLETADO")
        print(f"=" * 60)
        print(f"📊 Total procesadas: {report['processing_summary']['total_processed']}")
        print(f"✅ Exitosas: {report['processing_summary']['successful']}")
        print(f"❌ Rechazadas: {report['processing_summary']['rejected']}")
        print(f"📈 Tasa de éxito: {report['processing_summary']['success_rate']}")
        print(f"🎯 Listo para LoRA: {'SÍ' if report['ready_for_lora'] else 'NO'}")

        # Mostrar estadísticas RAW si hay
        if (
            report.get("raw_processing_stats")
            and report["raw_processing_stats"].get("raw_files_processed", 0) > 0
        ):
            raw_stats = report["raw_processing_stats"]
            print(f"\n📸 PROCESAMIENTO RAW:")
            print(f"   Archivos RAW: {raw_stats['raw_files_processed']}")
            print(f"   Archivos estándar: {raw_stats['standard_files_processed']}")
            print(f"   Éxito RAW: {raw_stats['raw_success_rate']*100:.1f}%")

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
