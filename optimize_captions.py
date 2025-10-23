#!/usr/bin/env python3
"""
optimize_captions.py - Convierte y optimiza captions para LoRA training
"""

import re
from pathlib import Path


def optimize_caption(caption_text, client_id):
    """
    Optimiza un caption para entrenamiento LoRA

    Reglas:
    1. Trigger word al inicio
    2. Eliminar caracteres problemáticos
    3. Eliminar redundancias
    4. Limitar longitud
    """

    # Limpiar caracteres problemáticos
    caption = caption_text.strip()
    caption = re.sub(r"\s+", " ", caption)  # Múltiples espacios → 1 espacio
    caption = re.sub(r"raw\s*:?\s*\d+", "", caption)  # Eliminar "raw :4"
    caption = caption.replace("    ", " ")  # Espacios múltiples

    # Separar en partes
    parts = [p.strip() for p in caption.split(",")]

    # Eliminar duplicados conservando orden
    seen = set()
    unique_parts = []
    for part in parts:
        part_lower = part.lower()
        if part_lower not in seen and part_lower:
            seen.add(part_lower)
            unique_parts.append(part)

    # Reconstruir caption optimizado
    # Asegurar que el trigger word esté al inicio
    if unique_parts and client_id.lower() not in unique_parts[0].lower():
        unique_parts.insert(0, client_id)

    # Limitar longitud (mantener solo las partes más importantes)
    if len(unique_parts) > 15:
        # Mantener: trigger word + características principales + calidad
        important_parts = unique_parts[:8] + [
            "detailed face",
            "high quality",
            "professional photography",
        ]
        unique_parts = [p for p in important_parts if p]

    optimized = ", ".join(unique_parts)

    return optimized


def convert_and_optimize_captions(dataset_dir, client_id):
    """Convierte .caption a .txt y optimiza el contenido"""

    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"❌ ERROR: No existe {dataset_path}")
        return False

    print("=" * 60)
    print("🔧 CONVERSIÓN Y OPTIMIZACIÓN DE CAPTIONS")
    print("=" * 60)

    # Buscar archivos .caption
    caption_files = list(dataset_path.glob("*.caption"))

    if not caption_files:
        print(f"\n❌ No se encontraron archivos .caption")

        # Verificar si ya son .txt
        txt_files = list(dataset_path.glob("*.txt"))
        if txt_files:
            print(f"\n✅ Ya existen {len(txt_files)} archivos .txt")

            optimize_existing = (
                input("\n¿Optimizar captions existentes? (s/n): ").strip().lower()
            )

            if optimize_existing in ["s", "si", "y", "yes"]:
                caption_files = txt_files
            else:
                return False
        else:
            return False

    print(f"\n📋 Archivos encontrados: {len(caption_files)}")
    print(f"📁 Ubicación: {dataset_path}")
    print(f"🎯 Cliente ID: {client_id}")

    # Mostrar ejemplo de optimización
    if caption_files:
        sample_file = caption_files[0]
        with open(sample_file, "r", encoding="utf-8") as f:
            original = f.read().strip()

        optimized = optimize_caption(original, client_id)

        print(f"\n📝 EJEMPLO DE OPTIMIZACIÓN:")
        print(f"\n   ANTES ({len(original)} caracteres):")
        print(f"   {original[:100]}...")
        print(f"\n   DESPUÉS ({len(optimized)} caracteres):")
        print(f"   {optimized}")

    # Confirmar
    print(f"\n¿Proceder con conversión y optimización?")
    confirm = input("Continuar? (s/n): ").strip().lower()

    if confirm not in ["s", "si", "y", "yes"]:
        print("❌ Proceso cancelado")
        return False

    # Procesar archivos
    print(f"\n🔄 Procesando archivos...")

    processed = 0
    errors = []

    for caption_file in caption_files:
        try:
            # Leer caption original
            with open(caption_file, "r", encoding="utf-8") as f:
                original_caption = f.read().strip()

            # Optimizar
            optimized_caption = optimize_caption(original_caption, client_id)

            # Determinar archivo de salida
            if caption_file.suffix == ".caption":
                txt_file = caption_file.with_suffix(".txt")
            else:
                txt_file = caption_file

            # Escribir caption optimizado
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(optimized_caption)

            # Si era .caption, eliminar el original
            if caption_file.suffix == ".caption" and txt_file != caption_file:
                caption_file.unlink()

            processed += 1

            if processed % 10 == 0:
                print(f"   Procesados: {processed}/{len(caption_files)}")

        except Exception as e:
            errors.append((caption_file.name, str(e)))

    # Resultados
    print(f"\n" + "=" * 60)
    print(f"📊 RESULTADOS")
    print("=" * 60)

    print(f"✅ Archivos procesados: {processed}")

    if errors:
        print(f"\n❌ Errores ({len(errors)}):")
        for filename, error in errors[:5]:
            print(f"   - {filename}: {error}")

    # Verificación final
    txt_files = list(dataset_path.glob("*.txt"))
    png_files = list(dataset_path.glob("*.png"))

    print(f"\n🔍 VERIFICACIÓN FINAL:")
    print(f"   Archivos .txt: {len(txt_files)}")
    print(f"   Archivos .png: {len(png_files)}")

    if len(txt_files) == len(png_files):
        print(f"   ✅ PERFECTO: Cada imagen tiene su caption")
    else:
        print(f"   ⚠️ Números no coinciden")

    # Mostrar ejemplos finales
    if txt_files:
        print(f"\n📝 EJEMPLOS DE CAPTIONS OPTIMIZADOS:")
        for i, txt_file in enumerate(txt_files[:3], 1):
            with open(txt_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()

            print(f"\n   {i}. {txt_file.name}")
            print(f"      {caption}")

    print(f"\n" + "=" * 60)
    print(f"✅ PROCESO COMPLETADO")
    print(f"\n💡 Ahora puedes iniciar el entrenamiento")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import sys

    # Configuración
    client_id = "Esoterico"  # AJUSTAR según tu cliente
    dataset_dir = r"E:\Proyectos\Avatar\clients\Esoterico\training_data\10_Esoterico"

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]

    if len(sys.argv) > 2:
        client_id = sys.argv[2]

    convert_and_optimize_captions(dataset_dir, client_id)
