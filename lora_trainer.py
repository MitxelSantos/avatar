#!/usr/bin/env python3
"""
lora_trainer.py
Módulo especializado para entrenamiento LoRA con Kohya_ss
Optimizado para SDXL en RTX 3050 6GB con máxima calidad
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
        self.training_configs = {
            "quality_optimized": {
                "name": "Máxima Calidad (6-8 horas)",
                "max_train_steps": 3000,
                "learning_rate": 0.00008,
                "dataset_repeats": 800,
                "save_every_n_steps": 300,
            },
            "balanced": {
                "name": "Equilibrado (4-6 horas)",
                "max_train_steps": 2000,
                "learning_rate": 0.0001,
                "dataset_repeats": 600,
                "save_every_n_steps": 400,
            },
            "fast": {
                "name": "Rápido (2-4 horas)",
                "max_train_steps": 1500,
                "learning_rate": 0.00012,
                "dataset_repeats": 400,
                "save_every_n_steps": 500,
            },
        }

    def configure_training(self, client_id, clients_dir):
        """Configura parámetros de entrenamiento LoRA"""
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

        print(f"\n⚙️ CONFIGURACIÓN ENTRENAMIENTO LoRA")
        print(f"Cliente: {client_id}")
        print(f"Dataset: {dataset_size} imágenes")
        print("=" * 50)

        # Mostrar opciones de configuración
        print("🎯 PERFILES DE ENTRENAMIENTO DISPONIBLES:")
        for key, config in self.training_configs.items():
            print(
                f"\n{list(self.training_configs.keys()).index(key) + 1}. {config['name']}"
            )
            print(f"   Steps: {config['max_train_steps']}")
            print(f"   Learning Rate: {config['learning_rate']}")
            print(f"   Dataset Repeats: {config['dataset_repeats']}")

            # Estimar tiempo
            estimated_hours = self.estimate_training_time(config, dataset_size)
            print(f"   Tiempo estimado: {estimated_hours:.1f} horas")

        print(f"\n4. ⚙️ Configuración personalizada")

        # Selección de perfil
        while True:
            try:
                choice = int(input("\nSelecciona perfil de entrenamiento (1-4): "))
                if choice == 4:
                    selected_config = self.create_custom_config(dataset_size)
                    break
                elif 1 <= choice <= 3:
                    config_key = list(self.training_configs.keys())[choice - 1]
                    selected_config = self.training_configs[config_key].copy()
                    break
                else:
                    print("❌ Opción inválida")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Generar configuración completa para SDXL
        full_config = self.generate_sdxl_config(
            client_id, selected_config, dataset_size
        )

        # Mostrar resumen de configuración
        self.show_config_summary(full_config, dataset_size)

        # Confirmar y guardar
        if input("\n¿Guardar esta configuración? (s/n): ").lower().startswith("s"):
            config_file = client_path / "training" / "lora_config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, "w") as f:
                json.dump(full_config, f, indent=2)

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

    def create_custom_config(self, dataset_size):
        """Crea configuración personalizada"""
        print(f"\n⚙️ CONFIGURACIÓN PERSONALIZADA")
        print("-" * 30)

        config = {}

        # Steps de entrenamiento
        default_steps = min(3000, dataset_size * 20)
        while True:
            try:
                steps = input(
                    f"Steps de entrenamiento (recomendado {default_steps}): "
                ).strip()
                config["max_train_steps"] = int(steps) if steps else default_steps
                if 500 <= config["max_train_steps"] <= 10000:
                    break
                else:
                    print("❌ Rango válido: 500-10000 steps")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Learning rate
        while True:
            try:
                lr = input("Learning rate (recomendado 0.00008): ").strip()
                config["learning_rate"] = float(lr) if lr else 0.00008
                if 0.00001 <= config["learning_rate"] <= 0.001:
                    break
                else:
                    print("❌ Rango válido: 0.00001-0.001")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Dataset repeats
        default_repeats = max(300, 1000 // dataset_size)
        while True:
            try:
                repeats = input(
                    f"Dataset repeats (recomendado {default_repeats}): "
                ).strip()
                config["dataset_repeats"] = int(repeats) if repeats else default_repeats
                if 100 <= config["dataset_repeats"] <= 2000:
                    break
                else:
                    print("❌ Rango válido: 100-2000")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Save frequency
        default_save = max(200, config["max_train_steps"] // 10)
        while True:
            try:
                save_freq = input(
                    f"Guardar cada N steps (recomendado {default_save}): "
                ).strip()
                config["save_every_n_steps"] = (
                    int(save_freq) if save_freq else default_save
                )
                break
            except ValueError:
                print("❌ Ingresa un número válido")

        config["name"] = "Configuración Personalizada"
        return config

    def generate_sdxl_config(self, client_id, base_config, dataset_size):
        """Genera configuración completa optimizada para SDXL en 6GB VRAM"""

        config = {
            # Información básica
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "profile_name": base_config["name"],
            "dataset_size": dataset_size,
            # Modelo base y VAE optimizados
            "model_config": {
                "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae": "madebyollin/sdxl-vae-fp16-fix",
                "v2": False,
                "v_parameterization": False,
                "clip_skip": 2,
            },
            # Configuración LoRA optimizada para 6GB VRAM
            "network_config": {
                "network_module": "networks.lora",
                "network_dim": 64,  # Reducido para 6GB
                "network_alpha": 32,
                "network_args": {
                    "conv_lora": True,
                    "conv_dim": 32,
                    "conv_alpha": 16,
                    "algo": "lora",
                },
            },
            # Optimizaciones críticas para 6GB VRAM
            "memory_optimizations": {
                "mixed_precision": "fp16",
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": 4,
                "cache_latents": True,
                "cache_text_encoder_outputs": True,
                "lowvram": True,
                "xformers_memory_efficient_attention": True,
            },
            # Configuración de entrenamiento
            "training_config": {
                "max_train_steps": base_config["max_train_steps"],
                "learning_rate": base_config["learning_rate"],
                "train_batch_size": 1,  # Fijo para 6GB
                "lr_scheduler": "cosine_with_restarts",
                "lr_warmup_steps": int(base_config["max_train_steps"] * 0.1),
                "optimizer_type": "AdamW8bit",  # Ahorra VRAM
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
            # Configuración del dataset
            "dataset_config": {
                "resolution": 1024,
                "bucket_resolution_steps": 64,
                "bucket_no_upscale": True,
                "min_bucket_reso": 512,
                "max_bucket_reso": 1024,
                "dataset_repeats": base_config["dataset_repeats"],
                "shuffle_caption": True,
                "caption_extension": ".txt",
                "keep_tokens": 1,
            },
            # Técnicas avanzadas para SDXL
            "advanced_config": {
                "noise_offset": 0.1,
                "adaptive_noise_scale": 0.00357,
                "multires_noise_iterations": 10,
                "multires_noise_discount": 0.1,
                "ip_noise_gamma": None,
                "debiased_estimation_loss": True,
            },
            # Configuración de guardado
            "save_config": {
                "save_every_n_steps": base_config["save_every_n_steps"],
                "save_model_as": "safetensors",
                "save_precision": "fp16",
                "output_name": f"{client_id}_avatar_lora",
                "max_checkpoints": 3,  # Solo mantener 3 checkpoints más recientes
            },
            # Configuración de muestras
            "sample_config": {
                "sample_every_n_steps": max(200, base_config["max_train_steps"] // 10),
                "sample_sampler": "euler_a",
                "sample_cfg_scale": 7.0,
                "sample_steps": 20,
                "sample_prompts": [
                    f"portrait of {client_id}, professional headshot, studio lighting, detailed face",
                    f"{client_id} smiling, natural lighting, high quality photography",
                    f"close-up of {client_id}, artistic portrait, dramatic lighting",
                ],
            },
            # Logging
            "logging_config": {
                "log_with": "tensorboard",
                "log_tracker_name": f"{client_id}_lora_training",
                "logging_dir": f"./logs/{client_id}",
                "log_level": "INFO",
            },
        }

        return config

    def show_config_summary(self, config, dataset_size):
        """Muestra resumen de la configuración"""
        print(f"\n📋 RESUMEN DE CONFIGURACIÓN:")
        print("=" * 40)
        print(f"Perfil: {config['profile_name']}")
        print(f"Cliente: {config['client_id']}")
        print(f"Dataset: {dataset_size} imágenes")

        print(f"\n🎯 PARÁMETROS PRINCIPALES:")
        print(f"   Steps totales: {config['training_config']['max_train_steps']:,}")
        print(f"   Learning rate: {config['training_config']['learning_rate']}")
        print(f"   Batch size: {config['training_config']['train_batch_size']}")
        print(f"   Dataset repeats: {config['dataset_config']['dataset_repeats']}")
        print(f"   Network dim: {config['network_config']['network_dim']}")

        print(f"\n🧠 OPTIMIZACIONES PARA 6GB VRAM:")
        print(f"   ✅ Mixed precision (fp16)")
        print(f"   ✅ Gradient checkpointing")
        print(f"   ✅ Cache latents y text encoder")
        print(f"   ✅ AdamW8bit optimizer")
        print(f"   ✅ XFormers memory efficient attention")

        # Estimación de tiempo
        estimated_time = self.estimate_training_time_detailed(config, dataset_size)
        print(f"\n⏱️ TIEMPO ESTIMADO:")
        print(f"   Duración total: {estimated_time['total_hours']:.1f} horas")
        print(f"   Steps por hora: ~{estimated_time['steps_per_hour']}")
        print(
            f"   Checkpoints: cada {config['save_config']['save_every_n_steps']} steps"
        )
        print(
            f"   Muestras: cada {config['sample_config']['sample_every_n_steps']} steps"
        )

    def estimate_training_time(self, config, dataset_size):
        """Estima tiempo de entrenamiento básico"""
        # Estimación aproximada para SDXL en RTX 3050
        steps_per_hour = 400  # Conservador para 6GB VRAM
        total_steps = config["max_train_steps"]
        return total_steps / steps_per_hour

    def estimate_training_time_detailed(self, config, dataset_size):
        """Estimación detallada de tiempo de entrenamiento"""
        # Factores que afectan velocidad en RTX 3050 6GB
        base_steps_per_hour = 400

        # Ajustes por configuración
        if config["dataset_config"]["resolution"] > 1024:
            base_steps_per_hour *= 0.7

        if config["training_config"]["train_batch_size"] > 1:
            base_steps_per_hour *= 0.8

        if config["memory_optimizations"]["gradient_accumulation_steps"] > 4:
            base_steps_per_hour *= 0.9

        total_steps = config["training_config"]["max_train_steps"]
        total_hours = total_steps / base_steps_per_hour

        return {
            "total_hours": total_hours,
            "steps_per_hour": int(base_steps_per_hour),
            "total_steps": total_steps,
        }

    def setup_training_environment(self, client_path):
        """Configura entorno de entrenamiento con Kohya_ss"""
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

        print(f"\n✅ ENTORNO CONFIGURADO CORRECTAMENTE")
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
        """Crea scripts de entrenamiento personalizados"""
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
        """Crea script principal de entrenamiento"""
        script_content = '''#!/usr/bin/env python3
"""
Script de entrenamiento LoRA optimizado para SDXL
Generado automáticamente por Avatar Pipeline
"""

import os
import sys
import json
import argparse
from pathlib import Path

def load_config(config_path):
    """Carga configuración de entrenamiento"""
    with open(config_path, 'r') as f:
        return json.load(f)

def build_training_command(config, dataset_path, output_path):
    """Construye comando de entrenamiento optimizado"""
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
        
        # Optimizaciones de memoria
        f"--mixed_precision={config['memory_optimizations']['mixed_precision']}",
        f"--gradient_accumulation_steps={config['memory_optimizations']['gradient_accumulation_steps']}",
        
        # Guardado
        f"--output_dir={output_path}",
        f"--output_name={config['save_config']['output_name']}",
        f"--save_model_as={config['save_config']['save_model_as']}",
        f"--save_every_n_steps={config['save_config']['save_every_n_steps']}",
        
        # Técnicas avanzadas
        f"--noise_offset={config['advanced_config']['noise_offset']}",
        f"--adaptive_noise_scale={config['advanced_config']['adaptive_noise_scale']}",
        
        # Logging
        f"--logging_dir={config['logging_config']['logging_dir']}",
        f"--log_with={config['logging_config']['log_with']}",
        
        # Flags de optimización
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--lowvram",
        "--xformers",
        "--bucket_no_upscale",
        "--shuffle_caption",
        "--caption_extension=.txt",
        "--keep_tokens=1",
        "--debiased_estimation_loss"
    ]
    
    # Agregar muestras si están configuradas
    if config.get('sample_config'):
        sample_dir = Path(output_path).parent / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        cmd.extend([
            f"--sample_every_n_steps={config['sample_config']['sample_every_n_steps']}",
            f"--sample_sampler={config['sample_config']['sample_sampler']}",
            f"--sample_folder={sample_dir}"
        ])
        
        # Crear archivo de prompts
        prompts_file = sample_dir / "sample_prompts.txt"
        with open(prompts_file, 'w') as f:
            for prompt in config['sample_config']['sample_prompts']:
                f.write(f"{prompt}\\n")
        
        cmd.append(f"--sample_prompts={prompts_file}")
    
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Construir comando
    cmd = build_training_command(config, args.dataset, args.output)
    
    # Ejecutar entrenamiento
    print("🚀 Iniciando entrenamiento LoRA...")
    print(f"Comando: {' '.join(cmd)}")
    
    import subprocess
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
'''

        with open(script_path, "w") as f:
            f.write(script_content)

    def create_monitoring_script(self, script_path):
        """Crea script de monitoreo de entrenamiento"""
        script_content = '''#!/usr/bin/env python3
"""
Monitor de entrenamiento LoRA
Genera reportes en tiempo real del progreso
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_training(log_dir, output_dir):
    """Monitorea progreso del entrenamiento"""
    print("📈 Iniciando monitoreo de entrenamiento...")
    
    last_step = 0
    start_time = time.time()
    
    while True:
        try:
            # Buscar logs más recientes
            log_files = list(Path(log_dir).glob("events.out.tfevents.*"))
            
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                # Buscar checkpoints
                checkpoints = list(Path(output_dir).glob("*.safetensors"))
                
                # Mostrar estado
                elapsed = time.time() - start_time
                print(f"\\r⏱️ Tiempo: {elapsed/3600:.1f}h | Checkpoints: {len(checkpoints)}", end="")
            
            time.sleep(30)  # Actualizar cada 30 segundos
            
        except KeyboardInterrupt:
            print("\\n🛑 Monitoreo detenido")
            break
        except Exception as e:
            print(f"\\n❌ Error en monitoreo: {e}")
            time.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python monitor_training.py <log_dir> <output_dir>")
        sys.exit(1)
    
    monitor_training(sys.argv[1], sys.argv[2])
'''

        with open(script_path, "w") as f:
            f.write(script_content)

    def create_batch_script(self, script_path, client_path):
        """Crea script batch para Windows"""
        batch_content = f"""@echo off
echo 🎯 Avatar Pipeline - Entrenamiento LoRA
echo Cliente: {client_path.name}
echo ================================================

cd /d "{self.kohya_path if self.kohya_path else './kohya_ss'}"

python "{client_path}/training/scripts/train_lora.py" ^
    --config "{client_path}/training/lora_config.json" ^
    --dataset "{client_path}/dataset_lora" ^
    --output "{client_path}/models"

echo ================================================
echo ✅ Entrenamiento completado
pause
"""

        with open(script_path, "w") as f:
            f.write(batch_content)

    def start_training(self, client_id, clients_dir):
        """Inicia entrenamiento LoRA"""
        client_path = clients_dir / client_id
        config_file = client_path / "training" / "lora_config.json"
        dataset_dir = client_path / "dataset_lora"

        if not config_file.exists():
            print(
                "❌ No hay configuración de entrenamiento. Configura primero los parámetros."
            )
            input("Presiona Enter para continuar...")
            return False

        if not dataset_dir.exists() or len(list(dataset_dir.glob("*.png"))) == 0:
            print("❌ No hay dataset preparado.")
            input("Presiona Enter para continuar...")
            return False

        print(f"\n🚀 INICIAR ENTRENAMIENTO LoRA")
        print(f"Cliente: {client_id}")
        print("=" * 50)

        # Cargar configuración para mostrar resumen
        with open(config_file, "r") as f:
            config = json.load(f)

        dataset_size = len(list(dataset_dir.glob("*.png")))

        print(f"📊 RESUMEN PRE-ENTRENAMIENTO:")
        print(f"   Dataset: {dataset_size} imágenes")
        print(f"   Steps totales: {config['training_config']['max_train_steps']:,}")
        print(f"   Learning rate: {config['training_config']['learning_rate']}")
        print(f"   Perfil: {config['profile_name']}")

        # Estimación de tiempo
        estimated = self.estimate_training_time_detailed(config, dataset_size)
        print(f"   Tiempo estimado: {estimated['total_hours']:.1f} horas")

        print(f"\n⚠️ IMPORTANTE:")
        print(f"   - El entrenamiento puede tomar varias horas")
        print(
            f"   - Se generarán muestras cada {config['sample_config']['sample_every_n_steps']} steps"
        )
        print(
            f"   - Los checkpoints se guardarán cada {config['save_config']['save_every_n_steps']} steps"
        )
        print(f"   - Puedes monitorear el progreso con TensorBoard")
        print(f"   - Puedes cancelar con Ctrl+C")

        confirm = input(f"\n🚀 ¿Iniciar entrenamiento? (s/n): ").lower().strip()
        if not confirm.startswith("s"):
            print("❌ Entrenamiento cancelado")
            input("Presiona Enter para continuar...")
            return False

        # Preparar directorios
        models_dir = client_path / "models"
        training_dir = client_path / "training"
        logs_dir = training_dir / "logs"

        models_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)

        print(f"\n🔧 PREPARANDO ENTRENAMIENTO...")

        # Verificar que Kohya_ss esté disponible
        if not self.kohya_path or not self.kohya_path.exists():
            print(
                f"❌ Kohya_ss no configurado. Ejecuta primero la configuración del entorno."
            )
            input("Presiona Enter para continuar...")
            return False

        print(f"✅ Kohya_ss: {self.kohya_path}")
        print(f"✅ Dataset: {dataset_dir}")
        print(f"✅ Modelos: {models_dir}")
        print(f"✅ Logs: {logs_dir}")

        # Iniciar entrenamiento
        print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
        print(f"📊 Progreso se guardará en: {logs_dir}")
        print(f"🎨 Muestras se guardarán en: {training_dir / 'samples'}")
        print(f"💾 Modelos se guardarán en: {models_dir}")

        success = self.execute_training(config, dataset_dir, models_dir, logs_dir)

        if success:
            print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
            print(f"📦 Modelos guardados en: {models_dir}")

            # Auto-exportar el mejor modelo
            self.auto_export_best_model(client_id, models_dir)
        else:
            print(f"\n❌ Entrenamiento falló. Revisa los logs para detalles.")

        input("Presiona Enter para continuar...")
        return success

    def execute_training(self, config, dataset_dir, models_dir, logs_dir):
        """Ejecuta el entrenamiento real con Kohya_ss"""
        try:
            # Construir comando de entrenamiento
            train_script = self.kohya_path / "train_network.py"

            if not train_script.exists():
                print(f"❌ Script de entrenamiento no encontrado: {train_script}")
                return False

            # Comando base
            cmd = [
                sys.executable,
                str(train_script),
                # Modelo y VAE
                f"--pretrained_model_name_or_path={config['model_config']['pretrained_model_name_or_path']}",
                f"--vae={config['model_config']['vae']}",
                # Dataset
                f"--train_data_dir={dataset_dir}",
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
                # Optimizaciones de memoria
                f"--mixed_precision={config['memory_optimizations']['mixed_precision']}",
                f"--gradient_accumulation_steps={config['memory_optimizations']['gradient_accumulation_steps']}",
                # Guardado
                f"--output_dir={models_dir}",
                f"--output_name={config['save_config']['output_name']}",
                f"--save_model_as={config['save_config']['save_model_as']}",
                f"--save_every_n_steps={config['save_config']['save_every_n_steps']}",
                # Técnicas avanzadas
                f"--noise_offset={config['advanced_config']['noise_offset']}",
                # Logging
                f"--logging_dir={logs_dir}",
                f"--log_with={config['logging_config']['log_with']}",
                # Flags de optimización
                "--gradient_checkpointing",
                "--cache_latents",
                "--cache_text_encoder_outputs",
                "--xformers",
                "--bucket_no_upscale",
                "--shuffle_caption",
                "--caption_extension=.txt",
                "--keep_tokens=1",
            ]

            # Ejecutar con output en tiempo real
            print(f"Ejecutando comando de entrenamiento...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.kohya_path,
            )

            # Mostrar output en tiempo real
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            return_code = process.poll()
            return return_code == 0

        except Exception as e:
            print(f"❌ Error durante entrenamiento: {str(e)}")
            return False

    def auto_export_best_model(self, client_id, models_dir):
        """Exporta automáticamente el mejor modelo"""
        model_files = list(models_dir.glob("*.safetensors"))

        if not model_files:
            print("⚠️ No se encontraron modelos para exportar")
            return

        # Seleccionar el modelo más reciente (generalmente el mejor)
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        # Exportar como modelo final
        final_name = f"{client_id}_avatar_lora_final.safetensors"
        final_path = models_dir / final_name

        shutil.copy2(latest_model, final_path)

        # Crear archivo de información
        model_info = {
            "client_id": client_id,
            "original_model": latest_model.name,
            "export_date": datetime.now().isoformat(),
            "model_size_mb": latest_model.stat().st_size / (1024 * 1024),
            "trigger_word": client_id,
            "usage_instructions": f"Use '{client_id}' as trigger word in your prompts",
            "recommended_weight": "0.7-1.0",
        }

        info_file = models_dir / f"{client_id}_avatar_lora_final.json"
        with open(info_file, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"✅ Modelo final exportado: {final_name}")
        print(f"📋 Info del modelo: {info_file}")
        print(f"💡 Trigger word: {client_id}")

        # Limpiar checkpoints antiguos (mantener solo los 2 más recientes + final)
        self.cleanup_old_checkpoints(models_dir, keep_count=2)

    def cleanup_old_checkpoints(self, models_dir, keep_count=2):
        """Limpia checkpoints antiguos automáticamente"""
        all_models = list(models_dir.glob("*.safetensors"))
        final_models = [m for m in all_models if "final" in m.name]
        checkpoint_models = [m for m in all_models if "final" not in m.name]

        if len(checkpoint_models) > keep_count:
            # Ordenar por fecha y mantener solo los más recientes
            sorted_checkpoints = sorted(
                checkpoint_models, key=lambda x: x.stat().st_mtime, reverse=True
            )
            to_delete = sorted_checkpoints[keep_count:]

            deleted_count = 0
            for model in to_delete:
                try:
                    model.unlink()
                    deleted_count += 1
                except Exception:
                    pass

            if deleted_count > 0:
                print(
                    f"🗑️ Limpieza automática: {deleted_count} checkpoints antiguos eliminados"
                )

    def show_training_progress(self, client_id, clients_dir):
        """Muestra progreso del entrenamiento"""
        client_path = clients_dir / client_id

        print(f"\n📈 PROGRESO DE ENTRENAMIENTO")
        print(f"Cliente: {client_id}")
        print("=" * 40)

        # Buscar logs y modelos
        logs_dir = client_path / "training" / "logs"
        models_dir = client_path / "models"
        samples_dir = client_path / "training" / "samples"

        # Estado del entrenamiento
        if models_dir.exists():
            model_files = list(models_dir.glob("*.safetensors"))
            print(f"💾 MODELOS GUARDADOS: {len(model_files)}")

            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                size_mb = latest_model.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(latest_model.stat().st_mtime)
                print(f"   Último modelo: {latest_model.name}")
                print(f"   Tamaño: {size_mb:.1f}MB")
                print(f"   Fecha: {mtime.strftime('%Y-%m-%d %H:%M')}")

        # Muestras generadas
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.png"))
            if sample_files:
                print(f"\n🎨 MUESTRAS GENERADAS: {len(sample_files)}")
                latest_sample = max(sample_files, key=lambda x: x.stat().st_mtime)
                print(f"   Última muestra: {latest_sample.name}")

        # Logs de TensorBoard
        if logs_dir.exists():
            tb_files = list(logs_dir.glob("**/events.out.tfevents.*", recursive=True))
            if tb_files:
                print(f"\n📊 LOGS DISPONIBLES:")
                print(f"   Archivos TensorBoard: {len(tb_files)}")
                print(f"   Para ver métricas: tensorboard --logdir {logs_dir}")

        input("\nPresiona Enter para continuar...")

    def generate_test_samples(self, client_id, clients_dir):
        """Genera muestras de prueba con modelo entrenado"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n🎨 GENERAR MUESTRAS DE PRUEBA")
        print(f"Cliente: {client_id}")
        print("=" * 40)

        # Verificar modelos disponibles
        if not models_dir.exists():
            print("❌ No se encontraron modelos entrenados")
            input("Presiona Enter para continuar...")
            return

        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No se encontraron modelos LoRA")
            input("Presiona Enter para continuar...")
            return

        # Mostrar modelos disponibles
        print("📦 MODELOS DISPONIBLES:")
        for i, model in enumerate(model_files, 1):
            size_mb = model.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(model.stat().st_mtime)
            print(
                f"{i:2d}. {model.name} ({size_mb:.1f}MB) - {mtime.strftime('%Y-%m-%d %H:%M')}"
            )

        # Este es un placeholder para generación real
        # En implementación completa integraría con diffusers o ComfyUI
        print(f"\n💡 FUNCIONALIDAD DE GENERACIÓN:")
        print(f"   Para generar muestras, usa el modelo LoRA en:")
        print(f"   - ComfyUI")
        print(f"   - Automatic1111")
        print(f"   - Diffusers (código Python)")
        print(f"\n🎯 Trigger word: {client_id}")
        print(f"📁 Modelos en: {models_dir}")

        input("Presiona Enter para continuar...")

    def manage_trained_models(self, client_id, clients_dir):
        """Gestiona modelos entrenados"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n📦 GESTIÓN DE MODELOS")
        print(f"Cliente: {client_id}")
        print("=" * 40)

        if not models_dir.exists():
            print("❌ No se encontraron modelos")
            input("Presiona Enter para continuar...")
            return

        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No se encontraron modelos LoRA")
            input("Presiona Enter para continuar...")
            return

        # Mostrar modelos con información
        total_size = 0
        print("📦 MODELOS DISPONIBLES:")

        for i, model in enumerate(sorted(model_files), 1):
            size_mb = model.stat().st_size / (1024 * 1024)
            total_size += size_mb
            mtime = datetime.fromtimestamp(model.stat().st_mtime)

            print(f"{i:2d}. {model.name}")
            print(
                f"     Tamaño: {size_mb:.1f}MB | Fecha: {mtime.strftime('%Y-%m-%d %H:%M')}"
            )

        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Total de modelos: {len(model_files)}")
        print(f"   Espacio usado: {total_size:.1f}MB")

        print(f"\n🔧 OPCIONES:")
        print("1. 📂 Abrir carpeta de modelos")
        print("2. 🗑️  Eliminar checkpoints antiguos")
        print("3. 📤 Información del modelo final")
        print("4. 🔙 Volver")

        choice = input("\nSelecciona opción (1-4): ").strip()

        if choice == "1":
            # Abrir carpeta
            try:
                os.startfile(str(models_dir))  # Windows
            except AttributeError:
                try:
                    subprocess.run(["open", str(models_dir)])  # macOS
                except FileNotFoundError:
                    subprocess.run(["xdg-open", str(models_dir)])  # Linux
            print(f"📂 Carpeta abierta: {models_dir}")

        elif choice == "2":
            self.cleanup_old_checkpoints(models_dir, keep_count=1)
            print("✅ Limpieza completada")

        elif choice == "3":
            final_models = [m for m in model_files if "final" in m.name]
            if final_models:
                final_model = final_models[0]
                info_file = final_model.with_suffix(".json")

                if info_file.exists():
                    with open(info_file, "r") as f:
                        info = json.load(f)

                    print(f"\n📋 INFORMACIÓN DEL MODELO FINAL:")
                    print(f"   Archivo: {final_model.name}")
                    print(f"   Trigger word: {info.get('trigger_word', client_id)}")
                    print(
                        f"   Peso recomendado: {info.get('recommended_weight', '0.7-1.0')}"
                    )
                    print(f"   Fecha creación: {info.get('export_date', 'N/A')}")
                else:
                    print(f"📦 Modelo final: {final_model.name}")
                    print(f"💡 Trigger word: {client_id}")
            else:
                print("❌ No se encontró modelo final")

        input("\nPresiona Enter para continuar...")
