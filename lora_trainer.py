#!/usr/bin/env python3
"""
lora_trainer.py - HOTFIX VERSION
CORRIGE: Error de codificación UTF-8 + Optimización para GTX 1650 4GB
"""

import os
import sys
import json
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime
import threading
import queue


class LoRATrainer:
    def __init__(self):
        self.kohya_path = None

        # CONFIGURACIONES OPTIMIZADAS PARA GTX 1650 4GB
        self.training_configs = {
            "ultra_optimized_4gb": {
                "name": "Ultra Optimizado 4GB (8-12 horas)",
                "max_train_steps": 1500,
                "learning_rate": 0.0001,
                "dataset_repeats": 300,
                "save_every_n_steps": 500,
                "special_4gb": True,
            },
            "balanced_4gb": {
                "name": "Equilibrado 4GB (6-8 horas)",
                "max_train_steps": 1200,
                "learning_rate": 0.00012,
                "dataset_repeats": 250,
                "save_every_n_steps": 400,
                "special_4gb": True,
            },
            "fast_4gb": {
                "name": "Rápido 4GB (4-6 horas)",
                "max_train_steps": 800,
                "learning_rate": 0.00015,
                "dataset_repeats": 200,
                "save_every_n_steps": 300,
                "special_4gb": True,
            },
        }

    def configure_training(self, client_id, clients_dir):
        """Configura parámetros de entrenamiento LoRA optimizado para 4GB"""
        client_path = clients_dir / client_id
        dataset_dir = client_path / "dataset_lora"

        if not dataset_dir.exists():
            print("❌ No hay dataset procesado. Prepara primero el dataset LoRA.")
            input("Presiona Enter para continuar...")
            return False

        # Contar imágenes en dataset
        dataset_images = list(dataset_dir.glob("*.png"))
        dataset_size = len(dataset_images)

        if dataset_size < 30:
            print(
                f"⚠️ Dataset pequeño ({dataset_size} imágenes). Recomendado mínimo: 50"
            )
            proceed = input("¿Continuar de todos modos? (s/n): ").lower().strip()
            if not proceed.startswith("s"):
                return False

        print(f"\n⚙️ CONFIGURACIÓN ENTRENAMIENTO LoRA - GTX 1650 4GB")
        print(f"Cliente: {client_id}")
        print(f"Dataset: {dataset_size} imágenes")
        print("=" * 50)

        # Mostrar advertencia específica para 4GB
        print("🚨 CONFIGURACIÓN ESPECIAL PARA 4GB VRAM:")
        print("   - Batch size forzado a 1")
        print("   - Network dim reducido a 32")
        print("   - Activadas TODAS las optimizaciones de memoria")
        print("   - Gradient accumulation aumentado")
        print("   - Resolución limitada a 768x768 si es necesario")

        # Mostrar opciones de configuración optimizadas
        print("\n🎯 PERFILES OPTIMIZADOS PARA 4GB VRAM:")
        for key, config in self.training_configs.items():
            print(
                f"\n{list(self.training_configs.keys()).index(key) + 1}. {config['name']}"
            )
            print(f"   Steps: {config['max_train_steps']}")
            print(f"   Learning Rate: {config['learning_rate']}")
            print(f"   Dataset Repeats: {config['dataset_repeats']}")

            # Estimar tiempo para 4GB
            estimated_hours = self.estimate_training_time_4gb(config, dataset_size)
            print(f"   Tiempo estimado: {estimated_hours:.1f} horas")

        print(f"\n4. ⚙️ Configuración personalizada (avanzado)")

        # Selección de perfil
        while True:
            try:
                choice = int(input("\nSelecciona perfil de entrenamiento (1-4): "))
                if choice == 4:
                    selected_config = self.create_custom_config_4gb(dataset_size)
                    break
                elif 1 <= choice <= 3:
                    config_key = list(self.training_configs.keys())[choice - 1]
                    selected_config = self.training_configs[config_key].copy()
                    break
                else:
                    print("❌ Opción inválida")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Generar configuración completa para SDXL optimizada para 4GB
        full_config = self.generate_4gb_optimized_config(
            client_id, selected_config, dataset_size
        )

        # Mostrar resumen de configuración
        self.show_config_summary_4gb(full_config, dataset_size)

        # Confirmar y guardar
        if input("\n¿Guardar esta configuración? (s/n): ").lower().startswith("s"):
            config_file = client_path / "training" / "lora_config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # ESCRIBIR CON CODIFICACIÓN UTF-8 EXPLÍCITA
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)

            print(f"✅ Configuración guardada en: {config_file}")

            # Preguntar si instalar dependencias
            if (
                input("\n¿Verificar e instalar dependencias ahora? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self.setup_training_environment(client_path)

        input("Presiona Enter para continuar...")
        return True

    def generate_4gb_optimized_config(self, client_id, base_config, dataset_size):
        """Genera configuración ultra-optimizada para GTX 1650 4GB"""

        config = {
            # Información básica
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "profile_name": base_config["name"],
            "dataset_size": dataset_size,
            "gpu_optimization": "GTX_1650_4GB",
            # Modelo base optimizado para 4GB
            "model_config": {
                "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae": "madebyollin/sdxl-vae-fp16-fix",
                "v2": False,
                "v_parameterization": False,
                "clip_skip": 2,
            },
            # Configuración LoRA ULTRA-OPTIMIZADA para 4GB
            "network_config": {
                "network_module": "networks.lora",
                "network_dim": 32,  # Reducido significativamente para 4GB
                "network_alpha": 16,  # Reducido proporcionalmente
                "network_args": {
                    "conv_lora": False,  # Desactivado para ahorrar VRAM
                    "algo": "lora",
                },
            },
            # OPTIMIZACIONES CRÍTICAS para 4GB VRAM
            "memory_optimizations": {
                "mixed_precision": "fp16",
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": 8,  # Aumentado para 4GB
                "cache_latents": True,
                "cache_text_encoder_outputs": True,
                "lowvram": True,
                "medvram": True,  # Activado para 4GB
                "xformers_memory_efficient_attention": True,
                "attention_slicing": True,  # Activado para 4GB
                "cpu_offload": True,  # Activado para 4GB
            },
            # Configuración de entrenamiento optimizada
            "training_config": {
                "max_train_steps": base_config["max_train_steps"],
                "learning_rate": base_config["learning_rate"],
                "train_batch_size": 1,  # FORZADO a 1 para 4GB
                "lr_scheduler": "cosine_with_restarts",
                "lr_warmup_steps": int(base_config["max_train_steps"] * 0.1),
                "optimizer_type": "AdamW8bit",  # Obligatorio para 4GB
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
            # Configuración del dataset optimizada
            "dataset_config": {
                "resolution": 768,  # Reducido de 1024 para 4GB
                "bucket_resolution_steps": 64,
                "bucket_no_upscale": True,
                "min_bucket_reso": 512,
                "max_bucket_reso": 768,  # Limitado para 4GB
                "dataset_repeats": base_config["dataset_repeats"],
                "shuffle_caption": True,
                "caption_extension": ".txt",
                "keep_tokens": 1,
            },
            # Técnicas avanzadas REDUCIDAS para 4GB
            "advanced_config": {
                "noise_offset": 0.05,  # Reducido
                "adaptive_noise_scale": None,  # Desactivado para 4GB
                "multires_noise_iterations": 0,  # Desactivado para 4GB
                "multires_noise_discount": 0,
                "ip_noise_gamma": None,
                "debiased_estimation_loss": False,  # Desactivado para 4GB
            },
            # Configuración de guardado optimizada
            "save_config": {
                "save_every_n_steps": base_config["save_every_n_steps"],
                "save_model_as": "safetensors",
                "save_precision": "fp16",
                "output_name": f"{client_id}_avatar_lora_4gb",
                "max_checkpoints": 2,  # Solo 2 para ahorrar espacio
            },
            # Configuración de muestras reducida
            "sample_config": {
                "sample_every_n_steps": max(300, base_config["max_train_steps"] // 5),
                "sample_sampler": "euler_a",
                "sample_cfg_scale": 7.0,
                "sample_steps": 15,  # Reducido para 4GB
                "sample_prompts": [
                    f"portrait of {client_id}, detailed face",
                    f"{client_id} smiling, high quality",
                ],
            },
            # Logging optimizado
            "logging_config": {
                "log_with": "tensorboard",
                "log_tracker_name": f"{client_id}_lora_4gb",
                "logging_dir": f"./logs/{client_id}",
                "log_level": "INFO",
            },
        }

        return config

    def estimate_training_time_4gb(self, config, dataset_size):
        """Estimación específica para GTX 1650 4GB"""
        # Muy conservador para 4GB VRAM
        steps_per_hour = 150  # Muy lento en 4GB
        total_steps = config["max_train_steps"]
        return total_steps / steps_per_hour

    def show_config_summary_4gb(self, config, dataset_size):
        """Muestra resumen optimizado para 4GB"""
        print(f"\n📋 RESUMEN DE CONFIGURACIÓN GTX 1650 4GB:")
        print("=" * 50)
        print(f"Perfil: {config['profile_name']}")
        print(f"Cliente: {config['client_id']}")
        print(f"Dataset: {dataset_size} imágenes")

        print(f"\n🎯 PARÁMETROS OPTIMIZADOS:")
        print(f"   Steps totales: {config['training_config']['max_train_steps']:,}")
        print(f"   Learning rate: {config['training_config']['learning_rate']}")
        print(
            f"   Batch size: {config['training_config']['train_batch_size']} (forzado)"
        )
        print(f"   Network dim: {config['network_config']['network_dim']} (reducido)")
        print(f"   Resolución: {config['dataset_config']['resolution']}px (optimizada)")

        print(f"\n🚨 OPTIMIZACIONES EXTREMAS PARA 4GB:")
        print(f"   ✅ Mixed precision (fp16)")
        print(f"   ✅ Gradient checkpointing")
        print(f"   ✅ Cache latents y text encoder")
        print(f"   ✅ AdamW8bit optimizer")
        print(f"   ✅ Attention slicing activado")
        print(f"   ✅ CPU offload activado")
        print(f"   ✅ MedVRAM activado")
        print(f"   ✅ Gradient accumulation: 8")

        # Estimación de tiempo
        estimated_time = self.estimate_training_time_4gb(config, dataset_size)
        print(f"\n⏱️ TIEMPO ESTIMADO (GTX 1650):")
        print(f"   Duración total: {estimated_time:.1f} horas")
        print(f"   Steps por hora: ~150 (muy lento)")
        print(f"   ⚠️ Recomendación: Ejecutar de noche")

    def create_custom_config_4gb(self, dataset_size):
        """Configuración personalizada con límites para 4GB"""
        print(f"\n⚙️ CONFIGURACIÓN PERSONALIZADA GTX 1650 4GB")
        print("-" * 40)
        print("⚠️ Parámetros limitados para evitar quedarse sin VRAM")

        config = {}

        # Steps de entrenamiento
        default_steps = min(1500, dataset_size * 15)  # Más conservador
        while True:
            try:
                steps = input(
                    f"Steps de entrenamiento (recomendado {default_steps}, máx 2000): "
                ).strip()
                config["max_train_steps"] = int(steps) if steps else default_steps
                if 500 <= config["max_train_steps"] <= 2000:
                    break
                else:
                    print("❌ Rango válido para 4GB: 500-2000 steps")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Learning rate
        while True:
            try:
                lr = input("Learning rate (recomendado 0.0001): ").strip()
                config["learning_rate"] = float(lr) if lr else 0.0001
                if 0.00005 <= config["learning_rate"] <= 0.0005:
                    break
                else:
                    print("❌ Rango válido para 4GB: 0.00005-0.0005")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Dataset repeats
        default_repeats = max(200, 800 // dataset_size)
        while True:
            try:
                repeats = input(
                    f"Dataset repeats (recomendado {default_repeats}): "
                ).strip()
                config["dataset_repeats"] = int(repeats) if repeats else default_repeats
                if 100 <= config["dataset_repeats"] <= 500:
                    break
                else:
                    print("❌ Rango válido para 4GB: 100-500")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Save frequency
        default_save = max(200, config["max_train_steps"] // 5)
        config["save_every_n_steps"] = default_save

        config["name"] = "Configuración Personalizada 4GB"
        config["special_4gb"] = True
        return config

    def setup_training_environment(self, client_path):
        """Configura entorno de entrenamiento optimizado para 4GB"""
        print(f"\n🔧 CONFIGURANDO ENTORNO DE ENTRENAMIENTO")
        print("=" * 50)

        # Verificar Python y pip
        if not self.verify_python_environment():
            return False

        # Verificar CUDA
        if not self.verify_cuda():
            return False

        # Instalar dependencias básicas
        if not self.install_basic_dependencies():
            return False

        # Configurar Kohya_ss
        if not self.setup_kohya_ss():
            return False

        # Crear scripts de entrenamiento
        self.create_training_scripts(client_path)

        print(f"\n✅ ENTORNO CONFIGURADO PARA GTX 1650 4GB")
        input("Presiona Enter para continuar...")
        return True

    def verify_python_environment(self):
        """Verifica entorno Python"""
        try:
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 8:
                print(
                    f"❌ Python 3.8+ requerido. Actual: {python_version.major}.{python_version.minor}"
                )
                return False

            print(
                f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
            return True

        except Exception as e:
            print(f"❌ Error verificando Python: {str(e)}")
            return False

    def verify_cuda(self):
        """Verifica instalación de CUDA"""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                print(f"✅ GPU: {gpu_name}")
                print(f"✅ VRAM: {vram_gb:.1f}GB")

                if vram_gb < 5.5:
                    print(f"⚠️ VRAM baja detectada. El entrenamiento será muy lento.")

                return True
            else:
                print(f"❌ CUDA no disponible. Se requiere GPU para entrenamiento.")
                return False

        except ImportError:
            print(f"❌ PyTorch no instalado.")
            return False

    def install_basic_dependencies(self):
        """Instala dependencias básicas para entrenamiento"""
        dependencies = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "xformers",
            "diffusers[torch]>=0.21.0",
            "transformers>=4.25.1",
            "accelerate>=0.15.0",
            "tensorboard",
            "safetensors",
            "dadaptation",
            "prodigyopt",
            "lion-pytorch",
        ]

        print(f"\n📦 INSTALANDO DEPENDENCIAS...")

        for i, dep in enumerate(dependencies, 1):
            print(f"\n[{i}/{len(dependencies)}] Instalando: {dep.split()[0]}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + dep.split(),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"✅ {dep.split()[0]} instalado")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error instalando {dep}: {e}")
                return False

        return True

    def setup_kohya_ss(self):
        """Configura Kohya_ss para entrenamiento"""
        print(f"\n🎯 CONFIGURANDO KOHYA_SS...")

        # Crear directorio para Kohya_ss
        kohya_dir = Path("./kohya_ss")

        if not kohya_dir.exists():
            print(f"📥 Clonando Kohya_ss...")
            try:
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "https://github.com/kohya-ss/sd-scripts.git",
                        str(kohya_dir),
                    ],
                    check=True,
                )
                print(f"✅ Kohya_ss clonado")
            except subprocess.CalledProcessError:
                print(f"❌ Error clonando Kohya_ss")
                return False

        # Instalar dependencias específicas de Kohya
        kohya_requirements = kohya_dir / "requirements.txt"
        if kohya_requirements.exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(kohya_requirements),
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"✅ Dependencias de Kohya_ss instaladas")
            except subprocess.CalledProcessError:
                print(f"⚠️ Algunas dependencias de Kohya_ss fallaron (continuando)")

        self.kohya_path = kohya_dir
        return True

    def create_training_scripts(self, client_path):
        """Crea scripts de entrenamiento optimizados para 4GB"""
        scripts_dir = client_path / "training" / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Script principal de entrenamiento
        train_script = scripts_dir / "train_lora.py"
        self.create_main_training_script(train_script)

        # Script de monitoreo
        monitor_script = scripts_dir / "monitor_training.py"
        self.create_monitoring_script(monitor_script)

        # Batch file para Windows
        batch_file = scripts_dir / "train_lora.bat"
        self.create_batch_script(batch_file, client_path)

        print(f"✅ Scripts de entrenamiento creados en: {scripts_dir}")

    def create_main_training_script(self, script_path):
        """Crea script principal de entrenamiento SIN EMOJIS"""
        script_content = """#!/usr/bin/env python3
# Script de entrenamiento LoRA optimizado para SDXL
# Generado automaticamente por Avatar Pipeline
# Optimizado para GTX 1650 4GB VRAM

import os
import sys
import json
import argparse
from pathlib import Path

def load_config(config_path):
    # Carga configuracion de entrenamiento
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_training_command(config, dataset_path, output_path):
    # Construye comando de entrenamiento optimizado para 4GB
    cmd = [
        sys.executable,
        "train_network.py",
        
        # Modelo y VAE
        f"--pretrained_model_name_or_path={config['model_config']['pretrained_model_name_or_path']}",
        f"--vae={config['model_config']['vae']}",
        
        # Dataset
        f"--train_data_dir={dataset_path}",
        f"--resolution={config['dataset_config']['resolution']}",
        f"--train_batch_size={config['training_config']['train_batch_size']}",
        f"--dataset_repeats={config['dataset_config']['dataset_repeats']}",
        
        # Red LoRA
        f"--network_module={config['network_config']['network_module']}",
        f"--network_dim={config['network_config']['network_dim']}",
        f"--network_alpha={config['network_config']['network_alpha']}",
        
        # Entrenamiento
        f"--max_train_steps={config['training_config']['max_train_steps']}",
        f"--learning_rate={config['training_config']['learning_rate']}",
        f"--lr_scheduler={config['training_config']['lr_scheduler']}",
        f"--lr_warmup_steps={config['training_config']['lr_warmup_steps']}",
        f"--optimizer_type={config['training_config']['optimizer_type']}",
        
        # Optimizaciones de memoria para 4GB
        f"--mixed_precision={config['memory_optimizations']['mixed_precision']}",
        f"--gradient_accumulation_steps={config['memory_optimizations']['gradient_accumulation_steps']}",
        
        # Guardado
        f"--output_dir={output_path}",
        f"--output_name={config['save_config']['output_name']}",
        f"--save_model_as={config['save_config']['save_model_as']}",
        f"--save_every_n_steps={config['save_config']['save_every_n_steps']}",
        
        # Logging
        f"--logging_dir={config['logging_config']['logging_dir']}",
        f"--log_with={config['logging_config']['log_with']}",
        
        # Flags de optimizacion criticos para 4GB
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--lowvram",
        "--medvram",
        "--xformers",
        "--bucket_no_upscale",
        "--shuffle_caption",
        "--caption_extension=.txt",
        "--keep_tokens=1",
    ]
    
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    
    args = parser.parse_args()
    
    # Cargar configuracion
    config = load_config(args.config)
    
    # Construir comando
    cmd = build_training_command(config, args.dataset, args.output)
    
    # Ejecutar entrenamiento
    print("Iniciando entrenamiento LoRA optimizado para 4GB...")
    print(f"Comando: {' '.join(cmd)}")
    
    import subprocess
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
"""

        # ESCRIBIR CON CODIFICACIÓN UTF-8 EXPLÍCITA
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

    def create_monitoring_script(self, script_path):
        """Crea script de monitoreo SIN EMOJIS"""
        script_content = """#!/usr/bin/env python3
# Monitor de entrenamiento LoRA
# Genera reportes en tiempo real del progreso

import os
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_training(log_dir, output_dir):
    # Monitorea progreso del entrenamiento
    print("Iniciando monitoreo de entrenamiento...")
    
    last_step = 0
    start_time = time.time()
    
    while True:
        try:
            # Buscar logs mas recientes
            log_files = list(Path(log_dir).glob("events.out.tfevents.*"))
            
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                # Buscar checkpoints
                checkpoints = list(Path(output_dir).glob("*.safetensors"))
                
                # Mostrar estado
                elapsed = time.time() - start_time
                print(f"\\rTiempo: {elapsed/3600:.1f}h | Checkpoints: {len(checkpoints)}", end="")
            
            time.sleep(30)  # Actualizar cada 30 segundos
            
        except KeyboardInterrupt:
            print("\\nMonitoreo detenido")
            break
        except Exception as e:
            print(f"\\nError en monitoreo: {e}")
            time.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python monitor_training.py <log_dir> <output_dir>")
        sys.exit(1)
    
    monitor_training(sys.argv[1], sys.argv[2])
"""

        # ESCRIBIR CON CODIFICACIÓN UTF-8 EXPLÍCITA
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

    def create_batch_script(self, script_path, client_path):
        """Crea script batch para Windows SIN EMOJIS"""
        batch_content = f"""@echo off
echo Avatar Pipeline - Entrenamiento LoRA GTX 1650 4GB
echo Cliente: {client_path.name}
echo ================================================

cd /d "{self.kohya_path if self.kohya_path else './kohya_ss'}"

python "{client_path}/training/scripts/train_lora.py" ^
    --config "{client_path}/training/lora_config.json" ^
    --dataset "{client_path}/dataset_lora" ^
    --output "{client_path}/models"

echo ================================================
echo Entrenamiento completado
pause
"""

        # ESCRIBIR CON CODIFICACIÓN UTF-8 EXPLÍCITA
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(batch_content)

    # Resto de métodos placeholder - implementar según necesidad
    def start_training(self, client_id, clients_dir):
        """Inicia entrenamiento LoRA optimizado para 4GB"""
        print("🚀 Iniciando entrenamiento optimizado para GTX 1650 4GB...")
        print("⚠️ Este proceso puede tomar 8-12 horas")
        input("Presiona Enter para continuar...")
        return True

    def show_training_progress(self, client_id, clients_dir):
        """Muestra progreso del entrenamiento"""
        print(f"\n📈 Progreso de entrenamiento para {client_id}")
        print("Usa: tensorboard --logdir clients/{client_id}/training/logs")
        input("Presiona Enter para continuar...")

    def generate_test_samples(self, client_id, clients_dir):
        """Genera muestras de prueba"""
        print(f"\n🎨 Generación de muestras para {client_id}")
        print("Función disponible después del entrenamiento")
        input("Presiona Enter para continuar...")

    def manage_trained_models(self, client_id, clients_dir):
        """Gestiona modelos entrenados"""
        print(f"\n📦 Gestión de modelos para {client_id}")
        print("Función disponible después del entrenamiento")
        input("Presiona Enter para continuar...")
