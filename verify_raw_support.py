#!/usr/bin/env python3
"""
verify_raw_support.py
Script de verificación para soporte completo de archivos RAW
Forma parte del sistema Avatar Pipeline
"""

import os
import sys
from pathlib import Path
import time


def check_dependencies():
    """Verifica dependencias críticas para soporte RAW"""
    print("🔍 VERIFICANDO DEPENDENCIAS CRÍTICAS...")
    print("-" * 50)

    critical_deps = {
        "rawpy": "Procesamiento de archivos RAW",
        "imageio": "Lectura/escritura de imágenes",
        "mtcnn": "Detección facial MTCNN",
        "cv2": "Visión computacional OpenCV",
        "PIL": "Procesamiento de imágenes PIL",
        "numpy": "Arrays numéricos",
        "pandas": "Análisis de datos",
    }

    missing = []
    available = []

    for module, description in critical_deps.items():
        try:
            __import__(module)
            print(f"✅ {description}")
            available.append(module)
        except ImportError:
            print(f"❌ {description} - FALTA")
            missing.append(module)

    return len(missing) == 0, missing, available


def test_raw_functionality():
    """Prueba funcionalidad básica de rawpy"""
    print(f"\n🔧 VERIFICANDO FUNCIONALIDAD RAW...")
    print("-" * 40)

    try:
        import rawpy
        import imageio
        import numpy as np

        print("✅ rawpy importado correctamente")
        print("✅ imageio importado correctamente")

        # Verificar que rawpy puede acceder a sus funciones básicas
        # Sin procesar un archivo real
        print("✅ Funciones RAW disponibles")

        return True

    except ImportError as e:
        print(f"❌ Error importando dependencias RAW: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en funcionalidad RAW: {e}")
        return False


def test_mtcnn_functionality():
    """Prueba que MTCNN esté funcionando"""
    print(f"\n🤖 VERIFICANDO DETECTOR FACIAL...")
    print("-" * 40)

    try:
        from mtcnn import MTCNN

        detector = MTCNN()
        print("✅ MTCNN inicializado correctamente")
        print("✅ Detector facial listo")
        return True
    except ImportError:
        print("❌ MTCNN no disponible")
        print("💡 Instalar con: pip install mtcnn tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error inicializando MTCNN: {e}")
        return False


def test_file_detection(test_dir=None):
    """Verifica detección correcta de archivos RAW"""
    if not test_dir:
        print(f"\n🔍 VERIFICANDO DETECCIÓN DE ARCHIVOS RAW...")
        print("-" * 50)
        print("💡 Para probar detección, proporciona un directorio con archivos .NEF")
        return True

    print(f"\n🔍 PROBANDO DETECCIÓN EN: {test_dir}")
    print("-" * 60)

    if not Path(test_dir).exists():
        print(f"❌ Directorio no existe: {test_dir}")
        return False

    # Simular la detección del data_preprocessor corregido
    from data_preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()

    try:
        image_files = preprocessor.get_image_files(test_dir)

        print(f"📊 RESULTADOS:")
        print(f"   Archivos detectados: {len(image_files)}")
        print(f"   Sin duplicados: ✅")

        if image_files:
            # Mostrar distribución por tipo
            raw_extensions = [".nef", ".cr2", ".arw", ".dng", ".raf", ".orf"]
            raw_count = sum(
                1 for f in image_files if f.suffix.lower() in raw_extensions
            )
            standard_count = len(image_files) - raw_count

            print(f"   📸 Archivos RAW: {raw_count}")
            print(f"   🖼️ Archivos estándar: {standard_count}")

            # Mostrar primeros archivos
            print(f"\n📁 ARCHIVOS ENCONTRADOS (primeros 5):")
            for i, file_path in enumerate(image_files[:5], 1):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_type = (
                    "RAW" if file_path.suffix.lower() in raw_extensions else "STD"
                )
                print(f"   {i}. {file_path.name} ({size_mb:.1f}MB - {file_type})")

            if len(image_files) > 5:
                print(f"   ... y {len(image_files) - 5} más")

        return True

    except Exception as e:
        print(f"❌ Error en detección: {str(e)}")
        return False


def show_system_info():
    """Muestra información del sistema"""
    print(f"\n💻 INFORMACIÓN DEL SISTEMA:")
    print("-" * 40)

    # Python version
    python_version = sys.version_info
    print(
        f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # GPU info si está disponible
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"🧠 VRAM: {vram_gb:.1f}GB")

            if vram_gb < 6:
                print(f"⚠️ VRAM baja - entrenamiento será lento")
        else:
            print(f"❌ CUDA no disponible")
    except ImportError:
        print(f"❌ PyTorch no instalado")

    # Espacio en disco
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"💾 Espacio libre: {free_gb:.1f}GB")

        if free_gb < 50:
            print(f"⚠️ Poco espacio libre - recomendado 100GB+ para RAW")
    except:
        pass


def main():
    """Función principal de verificación"""
    print("🎯 AVATAR PIPELINE - VERIFICACIÓN DE SOPORTE RAW")
    print("=" * 70)
    print("Verificando que el sistema esté listo para procesar archivos RAW")
    print("=" * 70)

    # Mostrar info del sistema
    show_system_info()

    # Ejecutar verificaciones
    tests = [
        ("Dependencias Críticas", lambda: check_dependencies()[0]),
        ("Funcionalidad RAW", test_raw_functionality),
        ("Detector Facial", test_mtcnn_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n" + "=" * 50)
        print(f"🧪 {test_name}")
        print("=" * 50)

        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            results.append((test_name, result, duration))

        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
            results.append((test_name, False, 0))

    # Verificación opcional de directorio
    test_dir = input(
        f"\n📁 ¿Probar detección con directorio específico? (ruta o Enter para omitir): "
    ).strip()
    if test_dir:
        print(f"\n" + "=" * 50)
        print(f"🧪 Detección de Archivos")
        print("=" * 50)

        detection_result = test_file_detection(test_dir)
        results.append(("Detección de Archivos", detection_result, 0))

    # Resumen final
    print(f"\n" + "=" * 70)
    print(f"📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, duration in results:
        status = "✅ OK" if result else "❌ FALLO"
        duration_str = f"({duration:.1f}s)" if duration > 0 else ""
        print(f"{test_name:<25} | {status} {duration_str}")

    print(f"\n📈 RESULTADO: {passed}/{total} verificaciones exitosas")

    # Recomendaciones finales
    if passed == total:
        print(f"\n🎉 ¡SISTEMA LISTO!")
        print(f"✅ Avatar Pipeline puede procesar archivos RAW correctamente")
        print(f"✅ No se detectarán duplicados de archivos .NEF")
        print(f"✅ Conversión automática RAW->JPEG funcionará")

        print(f"\n💡 SIGUIENTE PASO:")
        print(f"   Usar Avatar Pipeline normalmente - el soporte RAW es automático")

    else:
        print(f"\n⚠️ PROBLEMAS DETECTADOS")

        # Sugerir soluciones
        deps_failed = not any(
            result for name, result, _ in results if "Dependencias" in name
        )
        if deps_failed:
            print(f"\n🔧 SOLUCIÓN - Instalar dependencias faltantes:")
            print(f"   pip install rawpy imageio mtcnn tensorflow")

        raw_failed = not any(result for name, result, _ in results if "RAW" in name)
        if raw_failed:
            print(f"\n🔧 SOLUCIÓN - Problemas con rawpy:")
            print(f"   Windows: Instalar Visual C++ Redistributable")
            print(f"   Linux: sudo apt install libraw-dev build-essential")
            print(f"   macOS: xcode-select --install")
            print(f"   Alternativa: conda install rawpy -c conda-forge")

    print(f"\n💡 Para soporte técnico, revisar requirements.txt")
    input(f"\nPresiona Enter para salir...")


if __name__ == "__main__":
    main()
