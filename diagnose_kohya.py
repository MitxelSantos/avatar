#!/usr/bin/env python3
"""
diagnose_kohya.py - Script de diagnóstico para problemas de Kohya_ss
Ejecutar desde el directorio raíz del proyecto Avatar Pipeline
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_kohya_arguments():
    """Verifica qué argumentos están disponibles en train_network.py"""
    print("🔍 VERIFICANDO ARGUMENTOS DISPONIBLES EN KOHYA_SS")
    print("=" * 60)

    kohya_path = Path("kohya_ss")
    if not kohya_path.exists():
        print("❌ Directorio kohya_ss no encontrado")
        return []

    train_script = kohya_path / "train_network.py"
    if not train_script.exists():
        print("❌ train_network.py no encontrado")
        return []

    try:
        # Obtener ayuda completa
        result = subprocess.run(
            [sys.executable, "train_network.py", "--help"],
            cwd=kohya_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"❌ Error ejecutando --help: {result.stderr}")
            return []

        help_text = result.stdout
        arguments = []

        # Extraer todos los argumentos
        for line in help_text.split("\n"):
            line = line.strip()
            if line.startswith("--"):
                arg_name = line.split()[0]
                arguments.append(arg_name)

        print(f"✅ Encontrados {len(arguments)} argumentos disponibles")

        # Verificar argumentos problemáticos específicos
        problematic_args = [
            "--cache_text_encoder_outputs",
            "--lowvram",
            "--medvram",
            "--weight_decay",
        ]

        print(f"\n🔍 VERIFICANDO ARGUMENTOS PROBLEMÁTICOS:")
        for arg in problematic_args:
            status = "✅ DISPONIBLE" if arg in arguments else "❌ NO DISPONIBLE"
            print(f"   {arg}: {status}")

        return arguments

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def check_client_config():
    """Verifica configuración del cliente"""
    print(f"\n🔍 VERIFICANDO CONFIGURACIÓN DEL CLIENTE")
    print("=" * 50)

    clients_dir = Path("clients")
    if not clients_dir.exists():
        print("❌ Directorio clients no encontrado")
        return None

    # Buscar clientes disponibles
    client_dirs = [d for d in clients_dir.iterdir() if d.is_dir()]
    if not client_dirs:
        print("❌ No se encontraron clientes")
        return None

    print(f"📋 Clientes encontrados:")
    for i, client_dir in enumerate(client_dirs, 1):
        print(f"   {i}. {client_dir.name}")

    # Verificar configuración del primer cliente
    client_path = client_dirs[0]
    config_file = client_path / "training" / "lora_config.json"

    if not config_file.exists():
        print(f"❌ No se encontró configuración para {client_path.name}")
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        print(f"\n✅ Configuración encontrada para: {client_path.name}")
        print(f"   Preset: {config.get('preset_name', 'Unknown')}")
        print(
            f"   Steps: {config.get('training_config', {}).get('max_train_steps', 'Unknown')}"
        )
        print(f"   GPU: {config.get('detected_gpu', 'Unknown')}")

        return config

    except Exception as e:
        print(f"❌ Error leyendo configuración: {e}")
        return None


def check_gpu_status():
    """Verifica estado de la GPU"""
    print(f"\n🎮 VERIFICANDO GPU")
    print("=" * 25)

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU detectada: {gpu_name}")
            print(f"✅ VRAM disponible: {vram_gb:.1f}GB")
            print(f"✅ CUDA versión: {torch.version.cuda}")
            return True
        else:
            print("❌ CUDA no disponible")
            return False

    except ImportError:
        print("❌ PyTorch no instalado")
        return False


def generate_fixed_command():
    """Genera comando de entrenamiento corregido"""
    print(f"\n🔧 GENERANDO COMANDO CORREGIDO")
    print("=" * 35)

    # Argumentos básicos que siempre funcionan
    basic_args = [
        "python",
        "train_network.py",
        "--pretrained_model_name_or_path",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "--train_data_dir",
        "..\\clients\\TU_CLIENTE\\dataset_lora",
        "--output_dir",
        "..\\clients\\TU_CLIENTE\\models",
        "--network_module",
        "networks.lora",
        "--network_dim",
        "128",
        "--network_alpha",
        "64",
        "--max_train_steps",
        "2500",
        "--learning_rate",
        "0.0001",
        "--train_batch_size",
        "1",
        "--save_every_n_steps",
        "500",
        "--save_model_as",
        "safetensors",
        "--output_name",
        "TU_CLIENTE_avatar_lora",
        "--mixed_precision",
        "fp16",
        "--resolution",
        "1024",
        "--caption_extension",
        ".txt",
    ]

    print("📋 COMANDO BÁSICO (sin argumentos problemáticos):")
    print(" ".join(basic_args))

    print(f"\n💡 PASOS PARA EJECUTAR:")
    print("1. cd kohya_ss")
    print("2. Reemplaza TU_CLIENTE con el nombre real de tu cliente")
    print("3. Ejecuta el comando")


def main():
    """Función principal"""
    print("🔧 DIAGNÓSTICO DE KOHYA_SS - AVATAR PIPELINE")
    print("=" * 60)
    print("Script para diagnosticar problemas de entrenamiento LoRA")
    print()

    # 1. Verificar argumentos disponibles
    available_args = check_kohya_arguments()

    # 2. Verificar configuración del cliente
    client_config = check_client_config()

    # 3. Verificar GPU
    gpu_available = check_gpu_status()

    # 4. Generar comando corregido
    generate_fixed_command()

    print(f"\n" + "=" * 60)
    print("🎯 RESUMEN DEL DIAGNÓSTICO")
    print("=" * 60)

    if available_args:
        print("✅ Kohya_ss funcional")
    else:
        print("❌ Problema con Kohya_ss")

    if client_config:
        print("✅ Configuración de cliente encontrada")
    else:
        print("❌ Problema con configuración de cliente")

    if gpu_available:
        print("✅ GPU disponible")
    else:
        print("❌ Problema con GPU/CUDA")

    print(f"\n💡 PRÓXIMO PASO:")
    if all([available_args, client_config, gpu_available]):
        print("Aplicar el parche de compatibilidad y reintentar entrenamiento")
    else:
        print("Resolver problemas identificados antes del entrenamiento")


if __name__ == "__main__":
    main()
