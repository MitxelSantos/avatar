#!/usr/bin/env python3
"""
validate_images.py - Valida y repara imágenes con valores NaN/Inf
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil

def validate_and_fix_images(dataset_dir: Path, backup_dir: Path = None):
    """Valida todas las imágenes y las repara si tienen NaN/Inf"""
    
    dataset_dir = Path(dataset_dir)
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 VALIDANDO IMÁGENES EN: {dataset_dir}")
    print("=" * 70)
    
    image_files = sorted(dataset_dir.glob("*.png"))
    print(f"📊 Total de imágenes: {len(image_files)}\n")
    
    corrupted = []
    fixed = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # Cargar imagen
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32)
            
            # Verificar dimensiones
            if img.size != (1024, 1024):
                print(f"⚠️  {img_path.name}: Tamaño incorrecto {img.size}")
                corrupted.append((img_path, "wrong_size"))
                continue
            
            # Verificar NaN
            has_nan = np.isnan(img_array).any()
            
            # Verificar Inf
            has_inf = np.isinf(img_array).any()
            
            # Verificar rango
            min_val = img_array.min()
            max_val = img_array.max()
            out_of_range = min_val < 0 or max_val > 255
            
            if has_nan or has_inf or out_of_range:
                print(f"❌ {img_path.name}: CORRUPTA")
                print(f"   NaN: {has_nan}, Inf: {has_inf}")
                print(f"   Rango: [{min_val:.2f}, {max_val:.2f}]")
                
                corrupted.append((img_path, "nan_inf"))
                
                # Backup si se especificó
                if backup_dir:
                    backup_path = backup_dir / img_path.name
                    shutil.copy2(img_path, backup_path)
                    print(f"   💾 Backup: {backup_path}")
                
                # Intentar reparar
                print(f"   🔧 Intentando reparar...")
                
                # Reemplazar NaN/Inf con 0
                img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
                
                # Clipear a rango válido
                img_array = np.clip(img_array, 0, 255)
                
                # Convertir a uint8
                img_array = img_array.astype(np.uint8)
                
                # Guardar reparada
                repaired_img = Image.fromarray(img_array)
                repaired_img.save(img_path, "PNG", optimize=True)
                
                fixed.append(img_path)
                print(f"   ✅ Reparada y guardada")
            
            elif i % 10 == 0:
                print(f"✅ {i}/{len(image_files)}: OK")
                
        except Exception as e:
            print(f"❌ {img_path.name}: ERROR - {e}")
            corrupted.append((img_path, str(e)))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE VALIDACIÓN")
    print("=" * 70)
    print(f"✅ Imágenes validadas: {len(image_files)}")
    print(f"❌ Imágenes corruptas: {len(corrupted)}")
    print(f"🔧 Imágenes reparadas: {len(fixed)}")
    
    if len(corrupted) > 0:
        print(f"\n⚠️  IMÁGENES CON PROBLEMAS:")
        for img_path, reason in corrupted[:10]:
            print(f"   - {img_path.name}: {reason}")
        
        if len(corrupted) > 10:
            print(f"   ... y {len(corrupted) - 10} más")
    
    return len(corrupted) == 0


if __name__ == "__main__":
    dataset_path = Path("E:/Proyectos/Avatar/clients/Esoterico/training_data/10_Esoterico")
    backup_path = Path("E:/Proyectos/Avatar/clients/Esoterico/backup_corrupted")
    
    validate_and_fix_images(dataset_path, backup_path)