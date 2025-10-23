#!/usr/bin/env python3
"""
fix_captions.py - Convierte archivos .caption a .txt para Kohya_ss
"""

import os
from pathlib import Path
import shutil


def convert_captions_to_txt(dataset_dir):
    """Convierte todos los archivos .caption a .txt"""

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"❌ ERROR: No existe {dataset_path}")
        return False

    print("=" * 60)
    print("🔧 CONVERSIÓN DE CAPTIONS: .caption → .txt")
    print("=" * 60)

    # Buscar archivos .caption
    caption_files = list(dataset_path.glob("*.caption"))

    if not caption_files:
        print(f"\n❌ No se encontraron archivos .caption en:")
        print(f"   {dataset_path}")

        # Verificar si ya son .txt
        txt_files = list(dataset_path.glob("*.txt"))
        if txt_files:
            print(f"\n✅ Ya existen {len(txt_files)} archivos .txt")
            print(f"   No es necesario convertir")

        return False

    print(f"\n📋 Archivos .caption encontrados: {len(caption_files)}")
    print(f"📁 Ubicación: {dataset_path}")

    # Confirmar
    print(f"\n¿Convertir {len(caption_files)} archivos .caption a .txt?")
    print(f"Esto NO eliminará los archivos originales (se renombrarán)")

    confirm = input("\nContinuar? (s/n): ").strip().lower()

    if confirm not in ["s", "si", "y", "yes"]:
        print("❌ Conversión cancelada")
        return False

    # Convertir
    print(f"\n🔄 Convirtiendo archivos...")

    converted = 0
    errors = []

    for caption_file in caption_files:
        try:
            # Nuevo nombre con extensión .txt
            txt_file = caption_file.with_suffix(".txt")

            # Renombrar
            caption_file.rename(txt_file)

            converted += 1

            # Mostrar progreso cada 10 archivos
            if converted % 10 == 0:
                print(f"   Convertidos: {converted}/{len(caption_files)}")

        except Exception as e:
            errors.append((caption_file.name, str(e)))

    # Resultados
    print(f"\n" + "=" * 60)
    print(f"📊 RESULTADOS DE CONVERSIÓN")
    print("=" * 60)

    if converted > 0:
        print(f"✅ Archivos convertidos exitosamente: {converted}")

    if errors:
        print(f"\n❌ Errores ({len(errors)}):")
        for filename, error in errors[:5]:
            print(f"   - {filename}: {error}")
        if len(errors) > 5:
            print(f"   ... y {len(errors)-5} más")

    # Verificación final
    txt_files = list(dataset_path.glob("*.txt"))
    png_files = list(dataset_path.glob("*.png"))

    print(f"\n🔍 VERIFICACIÓN FINAL:")
    print(f"   Archivos .txt: {len(txt_files)}")
    print(f"   Archivos .png: {len(png_files)}")

    if len(txt_files) == len(png_files):
        print(f"   ✅ PERFECTO: Cada imagen tiene su caption")
    else:
        print(f"   ⚠️ ADVERTENCIA: Números no coinciden")

    # Mostrar ejemplos
    if txt_files:
        print(f"\n📝 EJEMPLOS DE CAPTIONS:")
        for i, txt_file in enumerate(txt_files[:3], 1):
            with open(txt_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            print(f"\n   {i}. {txt_file.name}")
            print(
                f"      → {caption[:80]}..."
                if len(caption) > 80
                else f"      → {caption}"
            )

    print(f"\n" + "=" * 60)
    print(f"✅ CONVERSIÓN COMPLETADA")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        # Ruta por defecto - ajusta según tu cliente
        dataset_dir = (
            r"E:\Proyectos\Avatar\clients\Esoterico\training_data\10_Esoterico"
        )

    convert_captions_to_txt(dataset_dir)
