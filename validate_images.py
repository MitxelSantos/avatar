#!/usr/bin/env python3
"""
validate_images_deep.py - Validación PROFUNDA para detectar causas de NaN
Analiza: formato, dimensiones, valores, estadísticas, canales, metadata
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import shutil
import json

def analyze_single_image(img_path: Path) -> dict:
    """Análisis exhaustivo de una sola imagen"""
    
    analysis = {
        "filename": img_path.name,
        "valid": True,
        "issues": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        # 1. CARGAR CON PIL
        img_pil = Image.open(img_path)
        analysis["stats"]["pil_mode"] = img_pil.mode
        analysis["stats"]["pil_size"] = img_pil.size
        analysis["stats"]["pil_format"] = img_pil.format
        
        # 2. CARGAR CON OPENCV
        img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_cv is None:
            analysis["valid"] = False
            analysis["issues"].append("OpenCV no pudo cargar la imagen")
            return analysis
        
        analysis["stats"]["cv_shape"] = img_cv.shape
        analysis["stats"]["cv_dtype"] = str(img_cv.dtype)
        analysis["stats"]["cv_channels"] = img_cv.shape[2] if len(img_cv.shape) == 3 else 1
        
        # 3. CONVERTIR A FLOAT PARA ANÁLISIS NUMÉRICO
        img_float = img_cv.astype(np.float32)
        
        # 4. VERIFICACIONES CRÍTICAS
        
        # Dimensiones
        if img_pil.size != (768, 768) and img_pil.size != (1024, 1024):
            analysis["warnings"].append(f"Dimensiones no estándar: {img_pil.size}")
        
        # NaN
        has_nan = np.isnan(img_float).any()
        if has_nan:
            nan_count = np.isnan(img_float).sum()
            nan_percent = (nan_count / img_float.size) * 100
            analysis["valid"] = False
            analysis["issues"].append(f"NaN detectados: {nan_count} ({nan_percent:.2f}%)")
            analysis["stats"]["nan_count"] = int(nan_count)
        
        # Inf
        has_inf = np.isinf(img_float).any()
        if has_inf:
            inf_count = np.isinf(img_float).sum()
            inf_percent = (inf_count / img_float.size) * 100
            analysis["valid"] = False
            analysis["issues"].append(f"Inf detectados: {inf_count} ({inf_percent:.2f}%)")
            analysis["stats"]["inf_count"] = int(inf_count)
        
        # Rango de valores
        min_val = float(img_float.min())
        max_val = float(img_float.max())
        mean_val = float(img_float.mean())
        std_val = float(img_float.std())
        
        analysis["stats"]["min"] = min_val
        analysis["stats"]["max"] = max_val
        analysis["stats"]["mean"] = mean_val
        analysis["stats"]["std"] = std_val
        
        if min_val < 0:
            analysis["issues"].append(f"Valores negativos detectados: min={min_val:.2f}")
            analysis["valid"] = False
        
        if max_val > 255:
            analysis["issues"].append(f"Valores > 255 detectados: max={max_val:.2f}")
            analysis["valid"] = False
        
        # Varianza extrema
        if std_val < 5:
            analysis["warnings"].append(f"Varianza muy baja (imagen casi uniforme): std={std_val:.2f}")
        
        # Brillo
        if mean_val < 30:
            analysis["warnings"].append(f"Imagen muy oscura: mean={mean_val:.2f}")
        elif mean_val > 225:
            analysis["warnings"].append(f"Imagen muy brillante: mean={mean_val:.2f}")
        
        # 5. ANÁLISIS POR CANAL (si es color)
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            for i, color in enumerate(['B', 'G', 'R']):
                channel = img_cv[:, :, i].astype(np.float32)
                analysis["stats"][f"channel_{color}_mean"] = float(channel.mean())
                analysis["stats"][f"channel_{color}_std"] = float(channel.std())
                
                if np.isnan(channel).any():
                    analysis["issues"].append(f"Canal {color} tiene NaN")
                    analysis["valid"] = False
        
        # 6. DETECCIÓN DE CORRUPCIÓN
        
        # Pixeles completamente negros o blancos
        black_pixels = np.sum(img_float == 0)
        white_pixels = np.sum(img_float == 255)
        total_pixels = img_float.size
        
        black_percent = (black_pixels / total_pixels) * 100
        white_percent = (white_pixels / total_pixels) * 100
        
        if black_percent > 80:
            analysis["warnings"].append(f"Imagen mayormente negra: {black_percent:.1f}%")
        if white_percent > 80:
            analysis["warnings"].append(f"Imagen mayormente blanca: {white_percent:.1f}%")
        
        # 7. TAMAÑO DE ARCHIVO
        file_size_kb = img_path.stat().st_size / 1024
        analysis["stats"]["file_size_kb"] = file_size_kb
        
        if file_size_kb < 50:
            analysis["warnings"].append(f"Archivo muy pequeño: {file_size_kb:.1f}KB")
        elif file_size_kb > 5000:
            analysis["warnings"].append(f"Archivo muy grande: {file_size_kb:.1f}KB")
        
        # 8. VERIFICAR METADATA EXIF
        try:
            exif_data = img_pil._getexif()
            if exif_data:
                analysis["stats"]["has_exif"] = True
        except:
            analysis["stats"]["has_exif"] = False
        
    except Exception as e:
        analysis["valid"] = False
        analysis["issues"].append(f"Error crítico: {str(e)}")
    
    return analysis


def repair_image(img_path: Path, backup_dir: Path, target_size=(768, 768)) -> bool:
    """Intenta reparar una imagen corrupta"""
    
    print(f"   🔧 Reparando: {img_path.name}")
    
    try:
        # Backup
        if backup_dir:
            backup_path = backup_dir / img_path.name
            shutil.copy2(img_path, backup_path)
            print(f"      💾 Backup: {backup_path.name}")
        
        # Cargar con OpenCV (más robusto)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"      ❌ No se pudo cargar con OpenCV")
            return False
        
        # Convertir a float para limpieza
        img_float = img.astype(np.float32)
        
        # Reemplazar NaN/Inf
        img_float = np.nan_to_num(img_float, nan=0.0, posinf=255.0, neginf=0.0)
        
        # Clipear a rango válido
        img_float = np.clip(img_float, 0, 255)
        
        # Convertir de vuelta a uint8
        img_clean = img_float.astype(np.uint8)
        
        # Redimensionar si es necesario
        if img_clean.shape[:2] != target_size:
            print(f"      📏 Redimensionando de {img_clean.shape[:2]} a {target_size}")
            img_clean = cv2.resize(img_clean, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Guardar
        cv2.imwrite(str(img_path), img_clean, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        
        # Verificar que la reparación funcionó
        verify = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        verify_float = verify.astype(np.float32)
        
        if np.isnan(verify_float).any() or np.isinf(verify_float).any():
            print(f"      ❌ Reparación falló - aún tiene NaN/Inf")
            return False
        
        print(f"      ✅ Reparada exitosamente")
        return True
        
    except Exception as e:
        print(f"      ❌ Error en reparación: {e}")
        return False


def validate_dataset_deep(dataset_dir: Path, backup_dir: Path = None, auto_repair: bool = False):
    """Validación profunda del dataset completo"""
    
    dataset_dir = Path(dataset_dir)
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🔬 VALIDACIÓN PROFUNDA DE DATASET")
    print(f"{'='*80}")
    print(f"📁 Directorio: {dataset_dir}")
    print(f"🔧 Auto-reparación: {'ACTIVADA' if auto_repair else 'DESACTIVADA'}")
    print(f"{'='*80}\n")
    
    image_files = sorted(dataset_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No se encontraron imágenes PNG")
        return False
    
    print(f"📊 Total de imágenes: {len(image_files)}\n")
    
    # Análisis de todas las imágenes
    all_analyses = []
    valid_count = 0
    invalid_count = 0
    warning_count = 0
    repaired_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i:3d}/{len(image_files)}] {img_path.name}")
        
        analysis = analyze_single_image(img_path)
        all_analyses.append(analysis)
        
        if analysis["valid"]:
            valid_count += 1
            if analysis["warnings"]:
                warning_count += 1
                print(f"   ⚠️  {len(analysis['warnings'])} advertencias")
            else:
                print(f"   ✅ OK")
        else:
            invalid_count += 1
            print(f"   ❌ INVÁLIDA: {len(analysis['issues'])} problemas")
            for issue in analysis["issues"]:
                print(f"      • {issue}")
            
            # Auto-reparación
            if auto_repair:
                if repair_image(img_path, backup_dir):
                    repaired_count += 1
    
    # RESUMEN DETALLADO
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN DE VALIDACIÓN")
    print(f"{'='*80}")
    print(f"✅ Válidas: {valid_count}/{len(image_files)} ({valid_count/len(image_files)*100:.1f}%)")
    print(f"⚠️  Con advertencias: {warning_count}/{len(image_files)}")
    print(f"❌ Inválidas: {invalid_count}/{len(image_files)} ({invalid_count/len(image_files)*100:.1f}%)")
    
    if auto_repair:
        print(f"🔧 Reparadas: {repaired_count}/{invalid_count}")
    
    # ESTADÍSTICAS AGREGADAS
    print(f"\n{'='*80}")
    print(f"📈 ESTADÍSTICAS DEL DATASET")
    print(f"{'='*80}")
    
    # Calcular estadísticas solo de imágenes válidas
    valid_analyses = [a for a in all_analyses if a["valid"]]
    
    if valid_analyses:
        means = [a["stats"]["mean"] for a in valid_analyses if "mean" in a["stats"]]
        stds = [a["stats"]["std"] for a in valid_analyses if "std" in a["stats"]]
        sizes_kb = [a["stats"]["file_size_kb"] for a in valid_analyses if "file_size_kb" in a["stats"]]
        
        if means:
            print(f"Brillo promedio: {np.mean(means):.1f} ± {np.std(means):.1f}")
        if stds:
            print(f"Contraste promedio: {np.mean(stds):.1f} ± {np.std(stds):.1f}")
        if sizes_kb:
            print(f"Tamaño archivo: {np.mean(sizes_kb):.1f}KB (min: {min(sizes_kb):.1f}, max: {max(sizes_kb):.1f})")
    
    # RECOMENDACIONES
    print(f"\n{'='*80}")
    print(f"💡 RECOMENDACIONES")
    print(f"{'='*80}")
    
    if invalid_count > 0:
        print(f"❌ Hay {invalid_count} imágenes con problemas CRÍTICOS")
        if not auto_repair:
            print(f"   Ejecuta con auto_repair=True para intentar repararlas")
        else:
            if repaired_count < invalid_count:
                print(f"   {invalid_count - repaired_count} imágenes NO pudieron ser reparadas")
                print(f"   Considera eliminarlas del dataset")
    
    if warning_count > 0:
        print(f"⚠️  Hay {warning_count} imágenes con advertencias")
        print(f"   Revisa manualmente si afectan el entrenamiento")
    
    if invalid_count == 0 and warning_count == 0:
        print(f"✅ Dataset completamente válido - listo para entrenar")
    
    # Guardar reporte JSON
    report_path = dataset_dir / "validation_report.json"
    report_data = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "total_images": len(image_files),
        "valid_images": valid_count,
        "invalid_images": invalid_count,
        "warning_images": warning_count,
        "repaired_images": repaired_count,
        "analyses": all_analyses
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Reporte guardado: {report_path}")
    
    return invalid_count == 0


if __name__ == "__main__":
    dataset_path = Path("E:/Proyectos/Avatar/clients/Esoterico/training_data/10_Esoterico")
    backup_path = Path("E:/Proyectos/Avatar/clients/Esoterico/backup_corrupted")
    
    # PRIMERA EJECUCIÓN: Solo validar
    print("\n🔍 FASE 1: VALIDACIÓN SIN REPARACIÓN")
    validate_dataset_deep(dataset_path, backup_path, auto_repair=False)
    
    input("\n⏸️  Presiona Enter para continuar con auto-reparación...")
    
    # SEGUNDA EJECUCIÓN: Validar y reparar
    print("\n🔧 FASE 2: VALIDACIÓN CON AUTO-REPARACIÓN")
    validate_dataset_deep(dataset_path, backup_path, auto_repair=True)