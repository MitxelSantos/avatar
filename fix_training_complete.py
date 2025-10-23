#!/usr/bin/env python3
"""
fix_training_complete.py - SOLUCIÓN COMPLETA EN UN SOLO SCRIPT
===============================================================

Este script hace TODO lo necesario para solucionar el problema de NaN:
1. Convierte .caption → .txt
2. Optimiza los captions
3. Redimensiona imágenes 1024→768
4. Ajusta configuración anti-NaN
5. Parchea data_preprocessor.py para futuro

Uso: python fix_training_complete.py
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import re


class TrainingFixer:
    def __init__(self, client_name="Esoterico"):
        self.client_name = client_name
        self.base_dir = Path(".")
        self.client_dir = self.base_dir / "clients" / client_name
        self.dataset_dir = None

        # Buscar subdirectorio del dataset
        training_data_parent = self.client_dir / "training_data"
        if training_data_parent.exists():
            subdirs = [d for d in training_data_parent.iterdir() if d.is_dir()]
            if subdirs:
                self.dataset_dir = subdirs[0]

    def print_header(self, title):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def step1_convert_captions(self):
        """Convierte .caption a .txt y optimiza contenido"""
        self.print_header("PASO 1: CONVERTIR Y OPTIMIZAR CAPTIONS")

        if not self.dataset_dir or not self.dataset_dir.exists():
            print("❌ No se encontró directorio de dataset")
            return False

        caption_files = list(self.dataset_dir.glob("*.caption"))
        txt_files = list(self.dataset_dir.glob("*.txt"))

        if not caption_files and txt_files:
            print(f"✅ Ya existen {len(txt_files)} archivos .txt")

            optimize = input("¿Optimizar captions existentes? (s/n): ").strip().lower()
            if optimize not in ["s", "si"]:
                return True

            caption_files = txt_files
        elif not caption_files:
            print("❌ No se encontraron archivos .caption ni .txt")
            return False

        print(f"📋 Procesando {len(caption_files)} captions...")

        for caption_file in caption_files:
            try:
                # Leer caption
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption = f.read().strip()

                # Optimizar
                caption = self._optimize_caption(caption)

                # Determinar archivo de salida
                txt_file = caption_file.with_suffix(".txt")

                # Escribir
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(caption)

                # Eliminar .caption si existe
                if caption_file.suffix == ".caption":
                    caption_file.unlink()

            except Exception as e:
                print(f"⚠️ Error en {caption_file.name}: {e}")

        txt_count = len(list(self.dataset_dir.glob("*.txt")))
        print(f"✅ Completado: {txt_count} archivos .txt listos")

        return True

    def _optimize_caption(self, caption):
        """Optimiza un caption eliminando problemas"""
        # Limpiar caracteres problemáticos
        caption = re.sub(r"\s+", " ", caption)
        caption = re.sub(r"raw\s*:?\s*\d+", "", caption)
        caption = caption.replace("    ", " ")

        # Separar y eliminar duplicados
        parts = [p.strip() for p in caption.split(",")]
        seen = set()
        unique_parts = []

        for part in parts:
            part_lower = part.lower()
            if part_lower and part_lower not in seen:
                seen.add(part_lower)
                unique_parts.append(part)

        # Asegurar trigger word al inicio
        if unique_parts and self.client_name.lower() not in unique_parts[0].lower():
            unique_parts.insert(0, self.client_name)

        # Limitar longitud
        if len(unique_parts) > 12:
            unique_parts = unique_parts[:9] + ["detailed face", "high quality"]

        return ", ".join(unique_parts)

    def step2_resize_images(self):
        """Redimensiona imágenes de 1024 a 768"""
        self.print_header("PASO 2: REDIMENSIONAR IMÁGENES 1024→768")

        if not self.dataset_dir:
            print("❌ No se encontró directorio de dataset")
            return False

        image_files = list(self.dataset_dir.glob("*.png"))

        if not image_files:
            print("❌ No se encontraron imágenes PNG")
            return False

        # Verificar tamaños
        sample = Image.open(image_files[0])

        if sample.size == (768, 768):
            print(f"✅ Las imágenes ya son 768x768")
            return True

        print(f"📋 Encontradas {len(image_files)} imágenes de {sample.size}")
        print(f"⚠️  Se redimensionarán a 768x768")

        # Crear backup
        backup_dir = Path(str(self.dataset_dir) + "_backup_1024")
        if not backup_dir.exists():
            print(f"💾 Creando backup en: {backup_dir.name}")
            shutil.copytree(
                self.dataset_dir,
                backup_dir,
                ignore=shutil.ignore_patterns("*.txt", "*.caption"),
            )

        # Redimensionar
        print(f"🔄 Redimensionando...")

        for i, img_file in enumerate(image_files, 1):
            try:
                img = Image.open(img_file)
                if img.size != (768, 768):
                    img_resized = img.resize((768, 768), Image.Resampling.LANCZOS)
                    img_resized.save(img_file, "PNG", optimize=True)

                if i % 20 == 0:
                    print(f"   {i}/{len(image_files)}")

            except Exception as e:
                print(f"⚠️ Error en {img_file.name}: {e}")

        print(f"✅ Completado: {len(image_files)} imágenes → 768x768")

        return True

    def step3_fix_config(self):
        """Ajusta lora_config.json con parámetros anti-NaN"""
        self.print_header("PASO 3: OPTIMIZAR CONFIGURACIÓN ANTI-NaN")

        config_file = self.client_dir / "training" / "lora_config.json"

        if not config_file.exists():
            print(f"❌ No se encontró {config_file}")
            return False

        # Cargar
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Aplicar cambios críticos
        changes = []

        # Learning Rate
        old_lr = config["training_config"]["learning_rate"]
        config["training_config"]["learning_rate"] = 0.00003
        changes.append(f"Learning Rate: {old_lr} → 0.00003")

        # Gradient Accumulation
        config["training_config"]["gradient_accumulation_steps"] = 4
        changes.append("Gradient Accumulation: 4")

        # Max Grad Norm
        config["training_config"]["max_grad_norm"] = 0.5
        changes.append("Max Grad Norm: 0.5")

        # Warmup Steps
        config["training_config"]["lr_warmup_steps"] = 100
        changes.append("Warmup Steps: 100")

        # Weight Decay
        config["training_config"]["weight_decay"] = 0.02
        changes.append("Weight Decay: 0.02")

        # Advanced config
        if "advanced_config" not in config:
            config["advanced_config"] = {}

        config["advanced_config"]["min_snr_gamma"] = 5
        config["advanced_config"]["noise_offset"] = 0.03
        changes.append("Min SNR Gamma: 5")
        changes.append("Noise Offset: 0.03")

        # Verificar resolución
        if config["dataset_config"]["resolution"] != 768:
            config["dataset_config"]["resolution"] = 768
            changes.append("Resolución: 768")

        # Guardar
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✅ Configuración actualizada:")
        for change in changes:
            print(f"   • {change}")

        return True

    def step4_patch_preprocessor(self):
        """Parchea data_preprocessor.py para futuro"""
        self.print_header("PASO 4: PARCHEAR data_preprocessor.py")

        file_path = self.base_dir / "data_preprocessor.py"

        if not file_path.exists():
            print(f"⚠️ No se encontró data_preprocessor.py")
            return False

        # Leer
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verificar si ya está parcheado
        if '.caption"' not in content:
            print(f"✅ data_preprocessor.py ya está actualizado")
            return True

        # Crear backup
        backup_path = file_path.with_suffix(".py.backup")
        if not backup_path.exists():
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"💾 Backup: {backup_path.name}")

        # Aplicar parche
        new_content = content.replace('.caption"', '.txt"')

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        count = content.count('.caption"')
        print(f"✅ Parcheado: {count} ocurrencias .caption → .txt")

        return True

    def verify_all(self):
        """Verifica que todo esté correcto"""
        self.print_header("VERIFICACIÓN FINAL")

        checks = {
            "Archivos .txt": False,
            "Imágenes 768x768": False,
            "Config optimizada": False,
            "data_preprocessor.py": False,
        }

        # Check 1: Captions
        if self.dataset_dir:
            txt_files = list(self.dataset_dir.glob("*.txt"))
            png_files = list(self.dataset_dir.glob("*.png"))
            checks["Archivos .txt"] = len(txt_files) == len(png_files) > 0

        # Check 2: Resolución
        if self.dataset_dir and png_files:
            sample = Image.open(png_files[0])
            checks["Imágenes 768x768"] = sample.size == (768, 768)

        # Check 3: Config
        config_file = self.client_dir / "training" / "lora_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

            lr_ok = config["training_config"]["learning_rate"] <= 0.0001
            res_ok = config["dataset_config"]["resolution"] == 768
            checks["Config optimizada"] = lr_ok and res_ok

        # Check 4: Preprocessor
        preprocessor = self.base_dir / "data_preprocessor.py"
        if preprocessor.exists():
            with open(preprocessor, "r") as f:
                content = f.read()
            checks["data_preprocessor.py"] = '.caption"' not in content

        # Mostrar resultados
        all_ok = True
        for check, status in checks.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {check}")
            if not status:
                all_ok = False

        print("\n" + "=" * 60)

        if all_ok:
            print("🎉 TODO LISTO PARA ENTRENAR")
            print("\nPróximo paso:")
            print("  python avatar_pipeline.py")
            print("  → Seleccionar cliente")
            print("  → Iniciar entrenamiento")
        else:
            print("⚠️ Algunos checks fallaron")

        print("=" * 60)

    def run_all(self):
        """Ejecuta todos los pasos"""
        print("\n" + "🔧" * 30)
        print("  SOLUCIÓN COMPLETA PROBLEMA NaN - TODO EN UNO")
        print("🔧" * 30)

        print(f"\nCliente: {self.client_name}")
        print(
            f"Dataset: {self.dataset_dir.name if self.dataset_dir else 'NO ENCONTRADO'}"
        )

        if not self.dataset_dir:
            print("\n❌ ERROR: No se pudo localizar el dataset")
            print("Verifica que exista: clients/{client}/training_data/")
            return False

        print(f"\n¿Proceder con la solución completa? (s/n): ", end="")
        confirm = input().strip().lower()

        if confirm not in ["s", "si", "y", "yes"]:
            print("❌ Cancelado")
            return False

        # Ejecutar todos los pasos
        success = True

        success &= self.step1_convert_captions()
        success &= self.step2_resize_images()
        success &= self.step3_fix_config()
        success &= self.step4_patch_preprocessor()

        # Verificar
        self.verify_all()

        return success


if __name__ == "__main__":
    import sys

    # Permitir especificar cliente
    client_name = sys.argv[1] if len(sys.argv) > 1 else "Esoterico"

    fixer = TrainingFixer(client_name)
    fixer.run_all()
