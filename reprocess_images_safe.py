#!/usr/bin/env python3
"""
reprocess_images_safe.py - Reprocesa imágenes con validación estricta
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def reprocess_image_safe(input_path: Path, output_path: Path) -> bool:
    """Reprocesa una imagen con validación estricta anti-NaN"""
    
    try:
        # Cargar con OpenCV (más robusto)
        img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"   ❌ No se pudo cargar: {input_path.name}")
            return False
        
        # Verificar que no haya NaN/Inf
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"   ⚠️  NaN/Inf detectado, limpiando...")
            img = np.nan_to_num(img, nan=0, posinf=255, neginf=0)
        
        # Asegurar que esté en rango válido
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Redimensionar si no es 1024x1024
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        
        # Convertir a PIL para guardar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Guardar con compresión PNG óptima
        pil_img.save(str(output_path), "PNG", optimize=True, compress_level=6)
        
        # Validar archivo guardado
        test_load = Image.open(output_path)
        test_array = np.array(test_load, dtype=np.float32)
        
        if np.isnan(test_array).any() or np.isinf(test_array).any():
            print(f"   ❌ Validación falló después de guardar")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def reprocess_all_images(dataset_dir: Path):
    """Reprocesa todas las imágenes del dataset"""
    
    dataset_dir = Path(dataset_dir)
    temp_dir = dataset_dir.parent / "temp_reprocessed"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n🔄 REPROCESANDO TODAS LAS IMÁGENES")
    print("=" * 70)
    
    image_files = sorted(dataset_dir.glob("*.png"))
    print(f"Total de imágenes: {len(image_files)}\n")
    
    success_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}")
        
        temp_output = temp_dir / img_path.name
        
        if reprocess_image_safe(img_path, temp_output):
            success_count += 1
            print(f"   ✅ OK")
        else:
            print(f"   ❌ FALLÓ - mantener original")
    
    # Si todo salió bien, reemplazar originales
    if success_count == len(image_files):
        print(f"\n✅ TODAS LAS IMÁGENES REPROCESADAS EXITOSAMENTE")
        print(f"🔄 Reemplazando originales...")
        
        for temp_file in temp_dir.glob("*.png"):
            original = dataset_dir / temp_file.name
            shutil.copy2(temp_file, original)
        
        shutil.rmtree(temp_dir)
        print(f"✅ Originales reemplazados")
    else:
        print(f"\n⚠️  SOLO {success_count}/{len(image_files)} se reprocesaron exitosamente")
        print(f"💡 Revisa las imágenes que fallaron manualmente")
        print(f"📁 Imágenes reprocesadas en: {temp_dir}")


if __name__ == "__main__":
    dataset_path = Path("E:/Proyectos/Avatar/clients/Esoterico/training_data/10_Esoterico")
    reprocess_all_images(dataset_path)