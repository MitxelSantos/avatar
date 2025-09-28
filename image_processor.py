#!/usr/bin/env python3
"""
image_processor.py - HOTFIX COMPLETO
CORRIGE: Parámetros rawpy + Bug DataFrame + Parámetros QC adaptativos
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from datetime import datetime
import json

# Soporte para archivos RAW
try:
    import rawpy
    import imageio
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False


class FaceProcessor:
    def __init__(self):
        self.detector = None
        self.init_mtcnn()
        
        # Extensiones soportadas
        self.supported_extensions = {
            'standard': ('.png', '.jpg', '.jpeg', '.tiff', '.tif'),
            'raw': ('.nef', '.cr2', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')
        }
        
    def init_mtcnn(self):
        """Inicializa detector MTCNN"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            print("✅ MTCNN detector loaded successfully")
        except ImportError:
            print("❌ MTCNN not available. Install: pip install mtcnn tensorflow")
            self.detector = None

    def convert_raw_on_demand(self, raw_path, temp_dir):
        """
        Convierte archivo RAW temporalmente para procesamiento
        VERSIÓN CORREGIDA - Compatible con rawpy estándar
        """
        temp_dir = Path(temp_dir)
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
                    # Removidos parámetros incompatibles
                )
            
            imageio.imwrite(str(temp_path), rgb, quality=95)
            return temp_path
            
        except Exception as e:
            print(f"   ❌ Error convirtiendo RAW: {str(e)}")
            print(f"   💡 Intentando configuración básica...")
            
            # Configuración de respaldo
            try:
                with rawpy.imread(str(raw_path)) as raw:
                    rgb = raw.postprocess()  # Usar configuración por defecto
                
                imageio.imwrite(str(temp_path), rgb, quality=90)
                return temp_path
                
            except Exception as e2:
                print(f"   ❌ Conversión falló completamente: {str(e2)}")
                return None

    def process_client_images(self, client_id, clients_dir, source_type='all', qc_params=None):
        """Procesa todas las imágenes de un cliente con soporte RAW completo"""
        client_path = clients_dir / client_id
        
        if not self.detector:
            print("❌ Detector facial no disponible")
            return False
        
        # Directorios fuente según tipo
        source_dirs = {}
        
        if source_type in ['all', 'mj'] and (client_path / "raw_mj").exists():
            source_dirs['mj'] = client_path / "raw_mj"
            
        if source_type in ['all', 'real'] and (client_path / "raw_real").exists():
            source_dirs['real'] = client_path / "raw_real"
            
        if not source_dirs:
            print(f"❌ No se encontraron directorios fuente para {source_type}")
            return False
        
        # Cargar metadata de fotos reales para conversiones RAW
        real_metadata = self.load_real_metadata(client_path)
        
        # Mostrar parámetros de calidad
        print(f"⚙️ PARÁMETROS DE CALIDAD:")
        for key, value in qc_params.items():
            readable_key = key.replace('_', ' ').title()
            print(f"   {readable_key}: {value}")
        print()
        
        # Procesar cada tipo de fuente
        all_results = []
        successful_count = 0
        failed_count = 0
        
        for img_type, source_dir in source_dirs.items():
            print(f"🔄 PROCESANDO {img_type.upper()}")
            print("-" * 30)
            
            # Obtener archivos de imagen
            image_files = self.get_processable_images(source_dir)
            
            if not image_files:
                print(f"❌ No se encontraron imágenes procesables en {source_dir}")
                continue
            
            print(f"📸 Encontradas {len(image_files)} imágenes para procesar")
            
            # Procesar cada imagen
            for i, image_file in enumerate(image_files, 1):
                print(f"\n📷 [{i:3d}/{len(image_files)}] {image_file.name}")
                
                # Determinar archivo fuente para procesamiento
                processing_file = self.get_processing_file(
                    image_file, real_metadata, client_path
                )
                
                if not processing_file:
                    print(f"   ❌ No se puede determinar archivo para procesar")
                    failed_count += 1
                    continue
                
                result = self.process_single_image(
                    image_path=str(processing_file),
                    original_path=str(image_file),
                    source_type=img_type,
                    output_dir=client_path / "processed",
                    qc_params=qc_params,
                    index=i
                )
                
                if result:
                    all_results.append(result)
                    if result['keep']:
                        successful_count += 1
                        print(f"   ✅ ACEPTADA - Confianza: {result['face_confidence']:.2f}")
                    else:
                        failed_count += 1
                        issues = ', '.join(result['qc_issues'])
                        print(f"   ❌ RECHAZADA - Motivos: {issues}")
                else:
                    failed_count += 1
                    print(f"   ❌ FALLÓ - No se pudo procesar")
                
                # Mostrar progreso cada 10 imágenes
                if i % 10 == 0:
                    print(f"\n📊 PROGRESO: {i}/{len(image_files)} procesadas | ✅{successful_count} ❌{failed_count}")
        
        # Generar reporte y mover archivos
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Generar reporte
            report = self.generate_processing_report(df, client_id)
            
            # Mover imágenes rechazadas (VERSIÓN CORREGIDA)
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

    def load_real_metadata(self, client_path):
        """Carga metadata de fotos reales para manejar conversiones RAW"""
        metadata_file = client_path / "metadata" / "real_metadata_mapping.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def get_processable_images(self, source_dir):
        """Obtiene imágenes procesables evitando duplicados"""
        all_extensions = self.supported_extensions['standard'] + self.supported_extensions['raw']
        
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

    def get_processing_file(self, original_file, real_metadata, client_path):
        """
        Determina qué archivo usar para procesamiento facial
        (original para estándar, conversión temporal para RAW)
        """
        filename = original_file.name
        
        # Verificar si es archivo RAW
        is_raw = original_file.suffix.lower() in [ext.lower() for ext in self.supported_extensions['raw']]
        
        if not is_raw:
            # Archivo estándar - usar directamente
            return original_file
        
        # Archivo RAW - buscar conversión temporal en metadata
        if filename in real_metadata:
            metadata = real_metadata[filename]
            temp_path = metadata.get('temp_conversion_path')
            
            if temp_path and Path(temp_path).exists():
                print(f"   📸 Usando conversión temporal para RAW: {Path(temp_path).name}")
                return Path(temp_path)
        
        # Si no hay conversión temporal, convertir on-the-fly
        if RAW_SUPPORT:
            print(f"   🔄 Convirtiendo RAW on-the-fly...")
            temp_dir = client_path / "temp_processing"
            temp_file = self.convert_raw_on_demand(original_file, temp_dir)
            return temp_file
        else:
            print(f"   ❌ Archivo RAW sin conversión disponible y rawpy no instalado")
            return None

    def process_single_image(self, image_path, original_path, source_type, output_dir, qc_params, index=1):
        """Procesa una sola imagen con detección facial completa - VERSIÓN CORREGIDA"""
        try:
            print(f"      🔍 Detectando rostro...")
            
            # Cargar imagen (ahora siempre será un formato estándar)
            img = cv2.imread(image_path)
            if img is None:
                print(f"      ❌ No se pudo cargar la imagen")
                # RETORNAR ESTRUCTURA CONSISTENTE INCLUSO EN FALLO
                return {
                    'original_path': str(original_path),
                    'processing_path': str(image_path),
                    'output_path': '',  # String vacía en lugar de None
                    'filename': '',     # String vacía en lugar de None
                    'source_type': source_type,
                    'face_confidence': 0.0,
                    'qc_issues': ['Could not load image'],
                    'keep': False,
                    'processed_date': datetime.now().isoformat(),
                    'file_size_kb': 0.0,
                    'image_dimensions': '0x0',
                    'is_raw_source': str(original_path) != str(image_path),
                }
            
            # Convertir a RGB para MTCNN
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detectar rostros
            results = self.detector.detect_faces(rgb_img)
            
            if not results:
                print(f"      ❌ No se detectó rostro")
                # RETORNAR ESTRUCTURA CONSISTENTE
                return {
                    'original_path': str(original_path),
                    'processing_path': str(image_path),
                    'output_path': '',
                    'filename': '',
                    'source_type': source_type,
                    'face_confidence': 0.0,
                    'qc_issues': ['No face detected'],
                    'keep': False,
                    'processed_date': datetime.now().isoformat(),
                    'file_size_kb': 0.0,
                    'image_dimensions': '0x0',
                    'is_raw_source': str(original_path) != str(image_path),
                }
            
            # Obtener mejor rostro (mayor confianza)
            face = max(results, key=lambda x: x['confidence'])
            confidence = face['confidence']
            
            print(f"      ✅ Rostro detectado - Confianza: {confidence:.2f}")
            
            # Verificar confianza mínima
            if confidence < qc_params['face_confidence_threshold']:
                print(f"      ❌ Confianza muy baja (mín: {qc_params['face_confidence_threshold']})")
                return {
                    'original_path': str(original_path),
                    'processing_path': str(image_path),
                    'output_path': '',
                    'filename': '',
                    'source_type': source_type,
                    'face_confidence': confidence,
                    'qc_issues': ['Low face confidence'],
                    'keep': False,
                    'processed_date': datetime.now().isoformat(),
                    'file_size_kb': 0.0,
                    'image_dimensions': '0x0',
                    'is_raw_source': str(original_path) != str(image_path),
                }
            
            print(f"      🔄 Recortando rostro...")
            
            # Recortar rostro en formato cuadrado
            cropped = self.crop_face_square(rgb_img, face, qc_params)
            if cropped is None:
                print(f"      ❌ Error en recorte")
                return {
                    'original_path': str(original_path),
                    'processing_path': str(image_path),
                    'output_path': '',
                    'filename': '',
                    'source_type': source_type,
                    'face_confidence': confidence,
                    'qc_issues': ['Crop failed'],
                    'keep': False,
                    'processed_date': datetime.now().isoformat(),
                    'file_size_kb': 0.0,
                    'image_dimensions': '0x0',
                    'is_raw_source': str(original_path) != str(image_path),
                }
            
            print(f"      🎨 Redimensionando a 1024x1024...")
            
            # Redimensionar y mejorar
            final_img = self.resize_and_enhance(cropped)
            
            # Generar nombre de archivo de salida
            output_filename = self.generate_output_filename(
                original_path, source_type, confidence, index
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
            
            # Metadata completa - ASEGURAR QUE TODOS LOS CAMPOS SEAN STRINGS VÁLIDAS
            metadata = {
                'original_path': str(original_path),
                'processing_path': str(image_path),
                'output_path': str(output_path),  # ASEGURAR string
                'filename': str(output_filename), # ASEGURAR string
                'source_type': str(source_type),
                'face_confidence': float(confidence),
                'face_box': face['box'],
                'face_keypoints': face.get('keypoints', {}),
                'qc_issues': qc_issues,
                'keep': len(qc_issues) == 0,
                'processed_date': datetime.now().isoformat(),
                'file_size_kb': float(os.path.getsize(output_path) / 1024),
                'image_dimensions': f"{final_img.size[0]}x{final_img.size[1]}",
                'is_raw_source': str(original_path) != str(image_path),
            }
            
            return metadata
            
        except Exception as e:
            print(f"      ❌ Error procesando: {str(e)}")
            # RETORNAR ESTRUCTURA CONSISTENTE INCLUSO EN EXCEPCIÓN
            return {
                'original_path': str(original_path),
                'processing_path': str(image_path),
                'output_path': '',
                'filename': '',
                'source_type': source_type,
                'face_confidence': 0.0,
                'qc_issues': [f'Processing error: {str(e)}'],
                'keep': False,
                'processed_date': datetime.now().isoformat(),
                'file_size_kb': 0.0,
                'image_dimensions': '0x0',
                'is_raw_source': str(original_path) != str(image_path),
            }
    
    def crop_face_square(self, img, face_info, qc_params):
        """Recorta rostro en formato cuadrado con padding"""
        x, y, w, h = face_info['box']
        
        # Calcular centro del rostro
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Determinar tamaño de recorte con padding
        face_size = max(w, h)
        crop_size = int(face_size * qc_params['face_padding_factor'])
        
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
            issues.append(f'Wrong size: {img.size}')
            print(f"      ❌ Tamaño incorrecto: {img.size}")
        
        # Verificar tamaño de archivo
        if os.path.exists(image_path):
            file_size_kb = os.path.getsize(image_path) / 1024
            if file_size_kb < qc_params['min_file_size_kb']:
                issues.append(f'File too small: {file_size_kb:.1f}KB')
                print(f"      ❌ Archivo muy pequeño: {file_size_kb:.1f}KB")
            elif file_size_kb > qc_params['max_file_size_mb'] * 1024:
                issues.append(f'File too large: {file_size_kb/1024:.1f}MB')
                print(f"      ❌ Archivo muy grande: {file_size_kb/1024:.1f}MB")
        
        # Convertir a escala de grises para análisis
        gray_array = np.array(img.convert('L'))
        
        # Verificar brillo
        avg_brightness = np.mean(gray_array)
        if avg_brightness < qc_params['min_brightness']:
            issues.append('Too dark')
            print(f"      ❌ Muy oscuro: {avg_brightness:.1f}")
        elif avg_brightness > qc_params['max_brightness']:
            issues.append('Too bright')
            print(f"      ❌ Muy brillante: {avg_brightness:.1f}")
        else:
            print(f"      ✅ Brillo OK: {avg_brightness:.1f}")
        
        # Verificar contraste
        contrast = np.std(gray_array)
        if contrast < qc_params['min_contrast']:
            issues.append('Low contrast')
            print(f"      ❌ Contraste bajo: {contrast:.1f}")
        else:
            print(f"      ✅ Contraste OK: {contrast:.1f}")
        
        # Verificar desenfoque (Laplacian variance)
        blur_score = cv2.Laplacian(gray_array, cv2.CV_64F).var()
        if blur_score < qc_params['blur_threshold']:
            issues.append(f'Blurry (score: {blur_score:.1f})')
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
        if source_type == 'mj':
            # Mantener información útil del nombre MJ original
            parts = original_name.split('_')
            if len(parts) > 3:
                descriptor = '_'.join(parts[1:4])  # Tomar partes del prompt
            else:
                descriptor = original_name
        else:
            # Para archivos reales, incluir info de formato
            original_ext = Path(original_path).suffix.upper()
            format_info = "RAW" if original_ext in ['.NEF', '.CR2', '.ARW', '.DNG', '.RAF', '.ORF'] else "STD"
            descriptor = f"real_{index:03d}_{format_info}"
        
        # Nombre final con confianza
        filename = f"{source_type}_{descriptor}_conf{confidence:.2f}.png"
        
        return filename

    def move_rejected_images(self, df, client_path):
        """Mueve imágenes rechazadas a carpeta separada - VERSIÓN CORREGIDA"""
        rejected_df = df[df['keep'] == False]
        
        if len(rejected_df) == 0:
            print(f"\n✅ No hay imágenes rechazadas que mover")
            return
        
        print(f"\n📦 MOVIENDO {len(rejected_df)} IMÁGENES RECHAZADAS...")
        
        rejected_dir = client_path / "rejected"
        rejected_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        for i, (_, row) in enumerate(rejected_df.iterrows(), 1):
            try:
                # VALIDAR QUE output_path sea una string válida
                output_path = row.get('output_path')
                filename = row.get('filename')
                
                # Verificar que los datos sean válidos
                if not output_path or not filename:
                    print(f"  [{i:3d}] ⚠️ Datos incompletos para fila {i}")
                    continue
                    
                # Verificar que output_path sea string
                if not isinstance(output_path, str):
                    print(f"  [{i:3d}] ⚠️ output_path inválido: {type(output_path)} - {output_path}")
                    continue
                    
                # Verificar que filename sea string
                if not isinstance(filename, str):
                    print(f"  [{i:3d}] ⚠️ filename inválido: {type(filename)} - {filename}")
                    continue
                
                src = Path(output_path)
                dst = rejected_dir / filename
                
                # Verificar que el archivo fuente existe
                if not src.exists():
                    print(f"  [{i:3d}] ⚠️ Archivo no existe: {src}")
                    continue
                
                # Manejar duplicados en destino
                if dst.exists():
                    timestamp = datetime.now().strftime("%H%M%S")
                    dst = rejected_dir / f"{dst.stem}_dup{timestamp}{dst.suffix}"
                
                # Mover archivo
                src.rename(dst)
                moved_count += 1
                print(f"  [{i:3d}] ✅ Movida: {filename}")
                
            except Exception as e:
                print(f"  [{i:3d}] ❌ Error moviendo imagen {i}: {str(e)}")
                # Continuar con la siguiente imagen en lugar de fallar
                continue
        
        print(f"\n📊 Resumen: {moved_count}/{len(rejected_df)} imágenes movidas exitosamente")

    def generate_processing_report(self, df, client_id):
        """Genera reporte completo del procesamiento"""
        total_processed = len(df)
        successful = len(df[df['keep'] == True])
        rejected = total_processed - successful
        
        # Breakdown por tipo de fuente
        by_source_type = {}
        if 'source_type' in df.columns:
            by_source_type = df[df['keep'] == True]['source_type'].value_counts().to_dict()
        
        # Conteo de archivos RAW procesados
        raw_processed = len(df[df.get('is_raw_source', False) == True]) if 'is_raw_source' in df.columns else 0
        
        # Análisis de problemas de QC
        all_issues = []
        for issues_list in df['qc_issues']:
            if issues_list:
                all_issues.extend(issues_list)
        
        from collections import Counter
        issues_summary = Counter(all_issues)
        
        # Estadísticas de confianza facial
        confidence_stats = {
            'mean': df['face_confidence'].mean(),
            'min': df['face_confidence'].min(),
            'max': df['face_confidence'].max(),
            'std': df['face_confidence'].std()
        }
        
        report = {
            "client_id": client_id,
            "processing_date": datetime.now().isoformat(),
            "processing_summary": {
                "total_processed": total_processed,
                "successful": successful,
                "rejected": rejected,
                "success_rate": f"{(successful/total_processed)*100:.1f}%" if total_processed > 0 else "0%",
                "raw_files_processed": raw_processed
            },
            "by_source_type": by_source_type,
            "qc_issues_summary": dict(issues_summary),
            "face_confidence_stats": confidence_stats,
            "ready_for_lora": successful >= 50,
            "lora_training_recommendations": {
                "total_images": successful,
                "recommended_steps": min(3000, successful * 25),
                "batch_size": 1 if successful < 100 else 2,
                "learning_rate": 0.0001
            }
        }
        
        return report
    
    def prepare_processed_dataset(self, df, client_path):
        """Prepara dataset de imágenes procesadas exitosamente"""
        accepted_df = df[df['keep'] == True].copy()
        
        if len(accepted_df) == 0:
            print(f"\n❌ No hay imágenes aceptadas para el dataset")
            return
        
        print(f"\n🎯 PREPARANDO DATASET PROCESADO CON {len(accepted_df)} IMÁGENES...")
        
        processed_dir = client_path / "processed"
        
        # Las imágenes ya están en processed/, solo necesitamos organizar metadata
        print(f"✅ {len(accepted_df)} imágenes listas en {processed_dir}")
        
        # Estadísticas por tipo
        if 'source_type' in accepted_df.columns:
            type_counts = accepted_df['source_type'].value_counts()
            for img_type, count in type_counts.items():
                percentage = (count / len(accepted_df)) * 100
                print(f"   {img_type.upper()}: {count} imágenes ({percentage:.1f}%)")
        
        # Estadísticas RAW
        if 'is_raw_source' in accepted_df.columns:
            raw_count = len(accepted_df[accepted_df['is_raw_source'] == True])
            if raw_count > 0:
                print(f"   📸 Procesados desde RAW: {raw_count} imágenes")
    
    def save_processing_metadata(self, df, report, client_path):
        """Guarda metadata completa del procesamiento"""
        metadata_dir = client_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Guardar CSV con toda la información
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = metadata_dir / f"processing_log_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Guardar reporte JSON
        report_file = metadata_dir / f"processing_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Actualizar resumen general
        summary_file = metadata_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 METADATA GUARDADA:")
        print(f"   CSV: {csv_file}")
        print(f"   Reporte: {report_file}")
        print(f"   Resumen: {summary_file}")
    
    def show_final_results(self, report):
        """Muestra resultados finales del procesamiento"""
        print(f"\n" + "="*60)
        print(f"🎉 PROCESAMIENTO FACIAL COMPLETADO")
        print(f"="*60)
        print(f"📊 Total procesadas: {report['processing_summary']['total_processed']}")
        print(f"✅ Exitosas: {report['processing_summary']['successful']}")
        print(f"❌ Rechazadas: {report['processing_summary']['rejected']}")
        print(f"📈 Tasa de éxito: {report['processing_summary']['success_rate']}")
        print(f"🎯 Listo para LoRA: {'SÍ' if report['ready_for_lora'] else 'NO'}")
        
        # Estadísticas RAW
        raw_processed = report['processing_summary'].get('raw_files_processed', 0)
        if raw_processed > 0:
            print(f"📸 Archivos RAW procesados: {raw_processed}")
        
        # Mostrar problemas más comunes
        if report['qc_issues_summary']:
            print(f"\n🔍 PROBLEMAS MÁS COMUNES:")
            for issue, count in list(report['qc_issues_summary'].items())[:5]:
                print(f"   {issue}: {count} imágenes")
        
        # Breakdown por tipo
        if report['by_source_type']:
            print(f"\n📋 DISTRIBUCIÓN POR TIPO:")
            for img_type, count in report['by_source_type'].items():
                percentage = (count / report['processing_summary']['successful']) * 100
                print(f"   {img_type.upper()}: {count} imágenes ({percentage:.1f}%)")
        
        # Estadísticas de confianza facial
        conf_stats = report['face_confidence_stats']
        print(f"\n📊 CONFIANZA FACIAL:")
        print(f"   Promedio: {conf_stats['mean']:.2f}")
        print(f"   Rango: {conf_stats['min']:.2f} - {conf_stats['max']:.2f}")
        
        print(f"\n💡 Las imágenes procesadas están listas para el siguiente paso")