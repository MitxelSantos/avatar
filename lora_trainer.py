#!/usr/bin/env python3
"""
lora_trainer.py - Entrenador LoRA REAL - VERSIÓN CORREGIDA
FIXES APLICADOS:
- Método _setup_kohya_ss agregado (faltaba)
- Argumentos problemáticos removidos de _build_training_command
- Manejo de encoding UTF-8 para Windows
- Optimizado para RTX 8GB VRAM
- Compatibilidad completa con Kohya_ss actual
"""

import os
import sys
import json
import subprocess
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Imports locales
from config import CONFIG, GPUProfile
from utils import (
    PipelineLogger,
    ProgressTracker,
    save_json_safe,
    load_json_safe,
    estimate_processing_time,
)


class LoRATrainer:
    """Entrenador LoRA profesional con entrenamiento REAL - VERSIÓN CORREGIDA"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.logger = PipelineLogger("LoRATrainer", self.config.logs_dir)
        self.kohya_path = None

        # GPU detectada automáticamente
        self.detected_gpu_profile = self.config.detect_gpu_profile()
        if self.detected_gpu_profile:
            self.logger.info(f"GPU detectada: {self.detected_gpu_profile.name}")
        else:
            self.logger.warning("No se pudo detectar GPU compatible")

        # Estado del entrenamiento
        self.training_state = {
            "is_training": False,
            "current_client": None,
            "start_time": None,
            "config_used": None,
            "process": None,
        }

    def configure_training(self, client_id: str, clients_dir: Path) -> bool:
        """Configura parámetros de entrenamiento LoRA con detección automática de hardware"""
        self.logger.info(f"Configurando entrenamiento LoRA para cliente: {client_id}")

        client_path = clients_dir / client_id
        dataset_dir = client_path / "dataset_lora"

        print(f"\n⚙️ CONFIGURANDO ENTRENAMIENTO LORA")
        print(f"Cliente: {client_id}")
        print("-" * 40)

        # Validaciones iniciales
        if not self._validate_dataset(dataset_dir):
            return False

        # Obtener información del dataset
        dataset_info = self._analyze_dataset(dataset_dir)
        if not dataset_info:
            return False

        # Detectar GPU y mostrar información
        gpu_info = self._get_gpu_info()
        self._display_gpu_info(gpu_info, dataset_info)

        # Mostrar presets disponibles según GPU detectada
        available_presets = self._get_available_presets()
        selected_preset = self._select_training_preset(available_presets, dataset_info)

        if not selected_preset:
            self.logger.info("Configuración cancelada por el usuario")
            return False

        # Generar configuración completa
        full_config = self._generate_training_config(
            client_id, selected_preset, dataset_info, gpu_info
        )

        # Mostrar resumen y confirmar
        if not self._confirm_configuration(full_config, dataset_info, gpu_info):
            return False

        # Guardar configuración
        if self._save_training_config(full_config, client_path):
            # Preguntar si instalar dependencias
            setup_env = (
                input("\n¿Verificar e instalar dependencias ahora? (s/n): ")
                .lower()
                .strip()
            )
            if setup_env.startswith("s"):
                success = self._setup_training_environment(client_path)
                if not success:
                    print(
                        "⚠️ Algunos componentes fallaron, pero la configuración se guardó"
                    )
                    print("💡 Puedes intentar el entrenamiento de todas formas")
                return True
            return True

        return False

    def start_training(self, client_id: str, clients_dir: Path) -> bool:
        """Inicia entrenamiento LoRA REAL con validación exhaustiva"""
        print(f"\n🚀 INICIANDO ENTRENAMIENTO LORA REAL")
        print(f"Cliente: {client_id}")
        print("=" * 50)

        self.logger.info(f"Iniciando entrenamiento LoRA REAL para cliente: {client_id}")

        client_path = clients_dir / client_id
        config_file = client_path / "training" / "lora_config.json"

        # PASO 1: Validación exhaustiva
        print(f"\n🔍 PASO 1: VALIDACIONES PREVIAS")
        print("-" * 35)

        validation_results = {}

        # 1.1 Verificar configuración
        print(f"📄 Verificando configuración...")
        if config_file.exists():
            print(f"   ✅ Archivo de configuración encontrado")
            validation_results["config"] = True
        else:
            print(f"   ❌ Archivo de configuración NO encontrado")
            validation_results["config"] = False

        # 1.2 Verificar dataset
        print(f"📊 Verificando dataset...")
        dataset_dir = client_path / "dataset_lora"
        if dataset_dir.exists():
            dataset_images = list(dataset_dir.glob("*.png"))
            dataset_captions = list(dataset_dir.glob("*.txt"))
            print(
                f"   ✅ Dataset encontrado: {len(dataset_images)} imágenes, {len(dataset_captions)} captions"
            )
            validation_results["dataset"] = len(dataset_images) >= 20
        else:
            print(f"   ❌ Dataset NO encontrado")
            validation_results["dataset"] = False

        # 1.3 Verificar PyTorch y CUDA
        print(f"🎮 Verificando PyTorch y CUDA...")
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   ✅ GPU disponible: {gpu_name} ({vram_gb:.1f}GB)")
                validation_results["gpu"] = True
            else:
                print(f"   ❌ CUDA no disponible")
                validation_results["gpu"] = False
        except ImportError:
            print(f"   ❌ PyTorch no disponible")
            validation_results["gpu"] = False

        # 1.4 Verificar Kohya_ss
        print(f"🔧 Verificando Kohya_ss...")
        if not self.kohya_path:
            print(f"   ⚠️ Kohya_ss no configurado, intentando auto-setup...")
            if self._setup_kohya_ss():
                print(f"   ✅ Kohya_ss configurado automáticamente")
                validation_results["kohya"] = True
            else:
                print(f"   ❌ Falló auto-configuración de Kohya_ss")
                validation_results["kohya"] = False
        else:
            if self.kohya_path.exists():
                train_script = self.kohya_path / "train_network.py"
                if train_script.exists():
                    print(f"   ✅ Kohya_ss encontrado y funcional")
                    validation_results["kohya"] = True
                else:
                    print(f"   ❌ Kohya_ss incompleto (falta train_network.py)")
                    validation_results["kohya"] = False
            else:
                print(f"   ❌ Kohya_ss path inválido: {self.kohya_path}")
                validation_results["kohya"] = False

        # RESUMEN DE VALIDACIONES
        print(f"\n📋 RESUMEN DE VALIDACIONES:")
        print("-" * 30)
        all_valid = True
        for check, result in validation_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check.title()}")
            if not result:
                all_valid = False

        if not all_valid:
            print(f"\n❌ PREREQUISITOS FALTANTES")
            self._show_validation_fixes(validation_results)
            return False

        # PASO 2: Cargar configuración
        print(f"\n🔍 PASO 2: CARGANDO CONFIGURACIÓN")
        print("-" * 35)

        try:
            training_config = load_json_safe(config_file, {}, self.logger)
            if not training_config:
                print(f"❌ Error cargando configuración de entrenamiento")
                input("Presiona Enter para continuar...")
                return False

            print(f"✅ Configuración cargada exitosamente")
            print(f"   Preset: {training_config.get('preset_name', 'Unknown')}")
            print(
                f"   Steps: {training_config.get('training_config', {}).get('max_train_steps', 'Unknown')}"
            )

        except Exception as e:
            print(f"❌ Error crítico cargando configuración: {str(e)}")
            self.logger.error(f"Error cargando configuración: {e}")
            input("Presiona Enter para continuar...")
            return False

        # PASO 3: Confirmación final
        print(f"\n🔍 PASO 3: CONFIRMACIÓN FINAL")
        print("-" * 30)

        self._display_training_info(training_config, client_path)

        confirmed = self._confirm_training_start(training_config)
        if not confirmed:
            print(f"🔙 Entrenamiento cancelado por el usuario")
            input("Presiona Enter para continuar...")
            return False

        # PASO 4: Ejecutar entrenamiento REAL
        print(f"\n🚀 PASO 4: EJECUTANDO ENTRENAMIENTO REAL")
        print("-" * 40)
        print(f"⚠️  ESTE ES EL ENTRENAMIENTO REAL - NO SIMULACIÓN")
        print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            result = self._execute_real_training(
                training_config, client_path, client_id
            )
            return result
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO EN ENTRENAMIENTO:")
            print(f"   {str(e)}")
            print(f"\n🔧 DEBUG INFO:")
            import traceback

            traceback.print_exc()
            input("Presiona Enter para continuar...")
            return False

    def _setup_kohya_ss(self) -> bool:
        """MÉTODO AGREGADO: Configura Kohya_ss automáticamente"""
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
                    capture_output=True,
                )
                print(f"✅ Kohya_ss clonado")
            except:
                print(f"❌ Error clonando Kohya_ss")
                return False

        train_script = kohya_dir / "train_network.py"
        if train_script.exists():
            print(f"✅ Kohya_ss configurado")
            self.kohya_path = kohya_dir
            return True
        else:
            print(f"❌ train_network.py no encontrado")
            return False

    def _execute_real_training(
        self, config: Dict, client_path: Path, client_id: str
    ) -> bool:
        """MÉTODO CORREGIDO: Ejecuta entrenamiento LoRA REAL usando Kohya_ss con encoding UTF-8"""

        # Actualizar estado
        self.training_state.update(
            {
                "is_training": True,
                "current_client": client_id,
                "start_time": datetime.now(),
                "config_used": config,
            }
        )

        try:
            # Crear directorios necesarios
            models_dir = client_path / "models"
            logs_dir = client_path / "training" / "logs"
            models_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Construir comando de entrenamiento
            print(f"🔧 Generando comando de entrenamiento...")

            dataset_dir = client_path / "dataset_lora"
            cmd = self._build_training_command(
                config, dataset_dir, models_dir, logs_dir
            )

            if not cmd:
                print(f"❌ Error generando comando de entrenamiento")
                return False

            # Mostrar comando (opcional para debug)
            print(f"📋 Comando de entrenamiento generado ({len(cmd)} argumentos)")

            # Cambiar al directorio de Kohya_ss
            original_cwd = os.getcwd()
            os.chdir(self.kohya_path)

            try:
                # CONFIGURAR ENCODING PARA WINDOWS - FIX CRÍTICO
                env = os.environ.copy()
                if os.name == "nt":  # Windows
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Silenciar warnings TensorFlow

                # Ejecutar entrenamiento con monitoreo
                print(f"\n🚀 INICIANDO ENTRENAMIENTO REAL...")
                print(f"📁 Directorio de trabajo: {self.kohya_path}")
                print(f"💾 Modelos se guardarán en: {models_dir}")
                print(f"📊 Logs en: {logs_dir}")
                print(f"\n⏳ El entrenamiento puede tomar varias horas...")
                print(f"💡 Para monitorear progreso, abre otra terminal y ejecuta:")
                print(f"   tensorboard --logdir {logs_dir}")

                # Ejecutar proceso de entrenamiento con encoding corregido
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env=env,  # Environment con UTF-8
                    encoding="utf-8",  # Forzar UTF-8
                    errors="replace",  # Reemplazar caracteres problemáticos
                )

                self.training_state["process"] = process

                # Monitorear progreso en tiempo real
                success = self._monitor_training_progress(process, models_dir, config)

                return success

            finally:
                # Siempre regresar al directorio original
                os.chdir(original_cwd)
                self.training_state["is_training"] = False

        except Exception as e:
            self.logger.error(f"Error en entrenamiento real: {e}")
            print(f"❌ Error ejecutando entrenamiento: {str(e)}")
            self.training_state["is_training"] = False
            return False

    def _build_training_command(
        self, config: Dict, dataset_dir: Path, output_dir: Path, logs_dir: Path
    ) -> List[str]:
        """
        MÉTODO CORREGIDO: Usa directamente la estructura Kohya_ss creada por data_preprocessor
        NO reorganiza archivos - asume estructura correcta ya existe
        Optimizado para RTX 8GB sin argumentos problemáticos
        """
        try:
            model_config = config["model_config"]
            network_config = config["network_config"]
            training_config = config["training_config"]
            dataset_config = config["dataset_config"]
            memory_opts = config["memory_optimizations"]
            save_config = config["save_config"]

            # USAR ESTRUCTURA KOHYA_SS EXISTENTE
            # dataset_dir apunta a clients/client_id/dataset_lora (estructura antigua)
            # Pero ahora usamos clients/client_id/training_data (estructura nueva)

            client_path = dataset_dir.parent  # clients/client_id/
            client_id = config["client_id"]

            # RUTA CORRECTA PARA KOHYA_SS
            training_data_parent = client_path / "training_data"
            training_data_subdir = training_data_parent / client_id

            # Verificar que la estructura existe
            if not training_data_subdir.exists():
                self.logger.error(
                    f"Estructura Kohya_ss no encontrada: {training_data_subdir}"
                )
                self.logger.info("Ejecuta 'Preparar dataset LoRA' primero")
                return []

            # Verificar que hay imágenes
            dataset_images = list(training_data_subdir.glob("*.png"))
            if not dataset_images:
                self.logger.error(
                    f"No se encontraron imágenes en: {training_data_subdir}"
                )
                return []

            self.logger.info(f"Usando estructura Kohya_ss: {training_data_parent}")
            self.logger.info(f"Imágenes encontradas: {len(dataset_images)}")

            # COMANDO OPTIMIZADO PARA RTX 8GB
            cmd = [
                sys.executable,
                "train_network.py",
                # ESTRUCTURA KOHYA_SS - DIRECTORIO PADRE
                "--train_data_dir",
                str(training_data_parent),  # ← training_data/ (padre)
                # Modelo base y VAE - ESENCIALES
                "--pretrained_model_name_or_path",
                model_config["pretrained_model_name_or_path"],
                "--vae",
                model_config["vae"],
                # Dataset - ESENCIALES
                "--resolution",
                str(dataset_config["resolution"]),
                "--train_batch_size",
                str(training_config["train_batch_size"]),
                "--dataset_repeats",
                str(dataset_config["dataset_repeats"]),
                # Red LoRA - OPTIMIZADO PARA RTX 8GB
                "--network_module",
                network_config["network_module"],
                "--network_dim",
                str(network_config["network_dim"]),  # 128 para RTX 8GB
                "--network_alpha",
                str(network_config["network_alpha"]),  # 64 para RTX 8GB
                # Entrenamiento - ESENCIALES
                "--max_train_steps",
                str(training_config["max_train_steps"]),
                "--learning_rate",
                str(training_config["learning_rate"]),
                "--lr_scheduler",
                training_config["lr_scheduler"],
                "--lr_warmup_steps",
                str(training_config["lr_warmup_steps"]),
                "--optimizer_type",
                training_config["optimizer_type"],
                # Precisión y optimizaciones - RTX 8GB
                "--mixed_precision",
                memory_opts["mixed_precision"],
                # Guardado - ESENCIALES
                "--output_dir",
                str(output_dir),
                "--output_name",
                save_config["output_name"],
                "--save_model_as",
                save_config["save_model_as"],
                "--save_every_n_steps",
                str(save_config["save_every_n_steps"]),
                "--save_precision",
                save_config["save_precision"],
                # Logging - ESENCIALES
                "--logging_dir",
                str(logs_dir),
                "--log_with",
                "tensorboard",
                # Configuraciones de dataset - KOHYA_SS
                "--caption_extension",
                ".txt",
                "--shuffle_caption",
                "--keep_tokens",
                "1",
                # Bucket settings - ACTIVAR PARA SUBDIRECTORIOS
                "--enable_bucket",
                "--bucket_no_upscale",
                "--min_bucket_reso",
                str(dataset_config["min_bucket_reso"]),
                "--max_bucket_reso",
                str(dataset_config["max_bucket_reso"]),
                # Optimizaciones básicas para RTX 8GB
                "--gradient_checkpointing",
                "--cache_latents",
                # OMITIDO: "--cache_text_encoder_outputs" - causaba error
            ]

            # ARGUMENTOS OPCIONALES SEGUROS
            optional_args = []

            # Gradient accumulation si es mayor a 1
            if training_config.get("gradient_accumulation_steps", 1) > 1:
                optional_args.extend(
                    [
                        "--gradient_accumulation_steps",
                        str(training_config["gradient_accumulation_steps"]),
                    ]
                )

            # Max grad norm si está definido
            if training_config.get("max_grad_norm"):
                optional_args.extend(
                    ["--max_grad_norm", str(training_config["max_grad_norm"])]
                )

            # Técnicas avanzadas para RTX 8GB
            advanced_config = config.get("advanced_config", {})
            if advanced_config.get("noise_offset"):
                optional_args.extend(
                    ["--noise_offset", str(advanced_config["noise_offset"])]
                )

            if advanced_config.get("adaptive_noise_scale"):
                optional_args.extend(
                    [
                        "--adaptive_noise_scale",
                        str(advanced_config["adaptive_noise_scale"]),
                    ]
                )

            if advanced_config.get("multires_noise_iterations"):
                optional_args.extend(
                    [
                        "--multires_noise_iterations",
                        str(advanced_config["multires_noise_iterations"]),
                    ]
                )

            if advanced_config.get("multires_noise_discount"):
                optional_args.extend(
                    [
                        "--multires_noise_discount",
                        str(advanced_config["multires_noise_discount"]),
                    ]
                )

            # Agregar argumentos opcionales
            cmd.extend(optional_args)

            # OMITIR ARGUMENTOS PROBLEMÁTICOS que causaban errores:
            # --lowvram          (causaba error)
            # --medvram          (causaba error)
            # --weight_decay     (formato incorrecto)
            # --cache_text_encoder_outputs (no soportado)

            self.logger.info(f"Comando Kohya_ss generado: {len(cmd)} argumentos")
            self.logger.info(f"Optimizado para RTX 8GB sin argumentos problemáticos")
            self.logger.info(
                f"Estructura de datos verificada: {len(dataset_images)} imágenes"
            )

            return cmd

        except Exception as e:
            self.logger.error(f"Error construyendo comando Kohya_ss: {e}")
            return []

    def _monitor_training_progress(
        self, process: subprocess.Popen, models_dir: Path, config: Dict
    ) -> bool:
        """Monitorea el progreso del entrenamiento en tiempo real con encoding seguro"""

        max_steps = config["training_config"]["max_train_steps"]
        save_every = config["save_config"]["save_every_n_steps"]

        print(f"\n📊 MONITOREANDO ENTRENAMIENTO")
        print(f"🎯 Steps objetivo: {max_steps:,}")
        print(f"💾 Guardado cada: {save_every} steps")
        print("-" * 50)

        last_step = 0
        last_checkpoint_time = time.time()

        try:
            # Leer output en tiempo real con manejo de encoding
            for line in process.stdout:
                try:
                    line = line.strip()
                except (UnicodeDecodeError, UnicodeError):
                    continue  # Saltar líneas con problemas de encoding

                if line:
                    # Buscar información de steps
                    if "step:" in line.lower() or "steps:" in line.lower():
                        step_match = self._extract_step_from_line(line)
                        if step_match and step_match > last_step:
                            last_step = step_match
                            progress = (last_step / max_steps) * 100

                            # Mostrar progreso cada 50 steps o cada minuto
                            current_time = time.time()
                            if (
                                last_step % 50 == 0
                                or current_time - last_checkpoint_time > 60
                            ):

                                elapsed = (
                                    current_time
                                    - self.training_state["start_time"].timestamp()
                                )
                                if last_step > 0:
                                    eta = (elapsed / last_step) * (
                                        max_steps - last_step
                                    )
                                    eta_str = self._format_time(eta)
                                else:
                                    eta_str = "Calculando..."

                                print(
                                    f"📈 Step {last_step:,}/{max_steps:,} ({progress:.1f}%) | ETA: {eta_str}"
                                )
                                last_checkpoint_time = current_time

                    # Mostrar logs importantes (con filtrado de caracteres)
                    if any(
                        keyword in line.lower()
                        for keyword in ["error", "warning", "saved", "checkpoint"]
                    ):
                        # Limpiar caracteres problemáticos para display
                        clean_line = "".join(char for char in line if ord(char) < 128)
                        print(f"📝 {clean_line}")

                    # Log todo al archivo (con manejo de encoding)
                    try:
                        self.logger.info(f"TRAINING: {line}")
                    except (UnicodeEncodeError, UnicodeError):
                        # Si hay problema de encoding, loggear version limpia
                        clean_line = "".join(char for char in line if ord(char) < 128)
                        self.logger.info(f"TRAINING: {clean_line}")

            # Esperar a que termine el proceso
            return_code = process.wait()

            if return_code == 0:
                print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")

                # Verificar modelos generados
                model_files = list(models_dir.glob("*.safetensors"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model_size = latest_model.stat().st_size / (1024 * 1024)

                    print(f"🧠 Modelo final: {latest_model.name}")
                    print(f"📦 Tamaño: {model_size:.1f}MB")
                    print(f"📁 Ubicación: {latest_model}")

                    # Registrar en historial
                    duration = datetime.now() - self.training_state["start_time"]
                    self._update_training_history(
                        models_dir.parent, config, duration, True
                    )

                    print(
                        f"⏱️ Tiempo total: {self._format_time(duration.total_seconds())}"
                    )

                else:
                    print(f"⚠️ Entrenamiento completado pero no se encontraron modelos")
                    return False

                return True
            else:
                print(f"\n❌ ENTRENAMIENTO FALLÓ")
                print(f"Código de salida: {return_code}")
                return False

        except KeyboardInterrupt:
            print(f"\n⚠️ ENTRENAMIENTO INTERRUMPIDO POR USUARIO")
            try:
                process.terminate()
                process.wait(timeout=10)
            except:
                process.kill()
            return False
        except Exception as e:
            print(f"\n❌ Error monitoreando entrenamiento: {e}")
            return False

    def _extract_step_from_line(self, line: str) -> Optional[int]:
        """Extrae número de step de una línea de log"""
        import re

        # Patrones comunes para detectar steps
        patterns = [
            r"step[:\s]+(\d+)",
            r"steps[:\s]+(\d+)",
            r"(\d+)/\d+",  # formato step/total
        ]

        for pattern in patterns:
            match = re.search(pattern, line.lower())
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue

        return None

    def _format_time(self, seconds: float) -> str:
        """Formatea tiempo en formato legible"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _show_validation_fixes(self, validation_results: Dict):
        """Muestra soluciones para problemas de validación"""
        fixes = []

        if not validation_results.get("config", True):
            fixes.append("Ejecutar opción 4: Configurar entrenamiento LoRA")
        if not validation_results.get("dataset", True):
            fixes.append("Ejecutar opción 3: Procesar imágenes")
        if not validation_results.get("gpu", True):
            fixes.append("Verificar instalación de CUDA")
        if not validation_results.get("kohya", True):
            fixes.append(
                "Instalar Kohya_ss: git clone https://github.com/kohya-ss/sd-scripts.git kohya_ss"
            )

        print(f"💡 SOLUCIONES:")
        for i, fix in enumerate(fixes, 1):
            print(f"   {i}. {fix}")

        input("Presiona Enter para continuar...")

    def _validate_dataset(self, dataset_dir: Path) -> bool:
        """Valida que el dataset esté listo"""
        if not dataset_dir.exists():
            self.logger.error("Dataset no encontrado")
            print("❌ No hay dataset procesado. Prepara primero el dataset LoRA.")
            input("Presiona Enter para continuar...")
            return False

        # Contar imágenes
        dataset_images = list(dataset_dir.glob("*.png"))
        if len(dataset_images) < 20:
            self.logger.warning(f"Dataset pequeño: {len(dataset_images)} imágenes")
            print(
                f"⚠️ Dataset pequeño ({len(dataset_images)} imágenes). Recomendado mínimo: 30"
            )
            proceed = input("¿Continuar de todos modos? (s/n): ").lower().strip()
            if not proceed.startswith("s"):
                return False

        return True

    def _analyze_dataset(self, dataset_dir: Path) -> Optional[Dict[str, Any]]:
        """Analiza el dataset y retorna información detallada"""
        try:
            all_images = list(dataset_dir.glob("*.png"))
            mj_images = [img for img in all_images if "_mj_" in img.name]
            real_images = [img for img in all_images if "_real_" in img.name]

            config_file = dataset_dir / "dataset_config.json"
            dataset_config = load_json_safe(config_file, {}, self.logger)

            info = {
                "total_images": len(all_images),
                "mj_images": len(mj_images),
                "real_images": len(real_images),
                "avatar_type": dataset_config.get("avatar_type", "unknown"),
                "distribution": dataset_config.get("balance_ratio", "unknown"),
                "has_captions": len(list(dataset_dir.glob("*.txt"))) == len(all_images),
                "avg_file_size_mb": (
                    sum(img.stat().st_size for img in all_images)
                    / len(all_images)
                    / (1024 * 1024)
                    if all_images
                    else 0
                ),
            }
            return info
        except Exception as e:
            self.logger.error(f"Error analizando dataset: {e}")
            return None

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información detallada de la GPU"""
        if self.detected_gpu_profile:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    return {
                        "profile": self.detected_gpu_profile,
                        "name": torch.cuda.get_device_name(0),
                        "vram_gb": gpu_props.total_memory / (1024**3),
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "multiprocessors": gpu_props.multi_processor_count,
                        "available": True,
                    }
            except ImportError:
                pass

        return {
            "profile": None,
            "name": "Unknown GPU",
            "vram_gb": 0,
            "available": False,
            "error": "No GPU detectada o PyTorch no disponible",
        }

    def _display_gpu_info(self, gpu_info: Dict, dataset_info: Dict):
        """Muestra información de GPU y dataset"""
        print(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
        if gpu_info.get("vram_gb"):
            print(f"💾 VRAM: {gpu_info['vram_gb']:.1f}GB")

        print(f"\n📊 DATASET:")
        print(f"   Total: {dataset_info['total_images']} imágenes")
        print(f"   🎨 MJ: {dataset_info['mj_images']}")
        print(f"   📷 Real: {dataset_info['real_images']}")

    def _get_available_presets(self) -> List[Dict[str, Any]]:
        """Obtiene presets disponibles según la GPU detectada"""
        presets = []
        for preset_key, preset_config in self.config.training_presets.items():
            preset_info = preset_config.copy()
            preset_info["key"] = preset_key
            preset_info["recommended"] = False

            if self.detected_gpu_profile:
                if (
                    preset_key == "balanced"
                    and "3050" in self.detected_gpu_profile.name.lower()
                ):
                    preset_info["recommended"] = True
                elif (
                    preset_key == "quality"
                    and self.detected_gpu_profile.vram_gb_min >= 8
                ):
                    preset_info["recommended"] = True
                elif (
                    preset_key == "quick" and self.detected_gpu_profile.vram_gb_min < 6
                ):
                    preset_info["recommended"] = True

            presets.append(preset_info)
        return presets

    def _select_training_preset(
        self, presets: List[Dict], dataset_info: Dict
    ) -> Optional[Dict]:
        """Permite al usuario seleccionar un preset de entrenamiento"""
        print(f"\n🎯 PRESETS DE ENTRENAMIENTO:")

        for i, preset in enumerate(presets, 1):
            recommended_mark = " 👈 RECOMENDADO" if preset.get("recommended") else ""
            print(f"\n{i}. {preset['name']}{recommended_mark}")
            print(f"   {preset['description']}")
            print(f"   Steps: {preset['max_train_steps']}")
            print(f"   Learning Rate: {preset['learning_rate']}")

            if self.detected_gpu_profile:
                gpu_multiplier = (
                    3600 / self.detected_gpu_profile.steps_per_hour_estimate
                )
                total_hours = preset["max_train_steps"] * gpu_multiplier
                print(f"   Tiempo estimado: {total_hours:.1f} horas")

        print(f"\n{len(presets) + 1}. ⚙️ Configuración personalizada")
        print(f"{len(presets) + 2}. 🔙 Cancelar")

        while True:
            try:
                choice = int(input(f"\nSelecciona preset (1-{len(presets) + 2}): "))
                if choice == len(presets) + 2:
                    return None
                elif choice == len(presets) + 1:
                    return self._create_custom_preset(dataset_info)
                elif 1 <= choice <= len(presets):
                    return presets[choice - 1]
                else:
                    print("❌ Opción inválida")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                return None

    def _create_custom_preset(self, dataset_info: Dict) -> Dict[str, Any]:
        """Crea preset personalizado"""
        print(f"\n⚙️ CONFIGURACIÓN PERSONALIZADA")
        print("-" * 40)

        preset = {"name": "Configuración Personalizada", "key": "custom"}

        # Steps
        default_steps = min(3000, dataset_info["total_images"] * 30)
        while True:
            try:
                steps_input = input(
                    f"Steps de entrenamiento (default {default_steps}): "
                ).strip()
                steps = int(steps_input) if steps_input else default_steps
                if 500 <= steps <= 5000:
                    preset["max_train_steps"] = steps
                    break
                else:
                    print("❌ Rango válido: 500-5000 steps")
            except ValueError:
                print("❌ Ingresa un número válido")

        # Learning rate
        while True:
            try:
                lr_input = input("Learning rate (default 0.0001): ").strip()
                lr = float(lr_input) if lr_input else 0.0001
                if 0.00005 <= lr <= 0.0005:
                    preset["learning_rate"] = lr
                    break
                else:
                    print("❌ Rango válido: 0.00005-0.0005")
            except ValueError:
                print("❌ Ingresa un número válido")

        preset["save_every_n_steps"] = max(200, preset["max_train_steps"] // 6)
        preset["description"] = (
            f"Configuración personalizada - {preset['max_train_steps']} steps"
        )
        return preset

    def _generate_training_config(
        self, client_id: str, preset: Dict, dataset_info: Dict, gpu_info: Dict
    ) -> Dict[str, Any]:
        """Genera configuración completa de entrenamiento"""
        gpu_profile = gpu_info.get("profile") or self.config.gpu_profiles["low_end"]

        dataset_repeats = max(
            50,
            min(
                preset.get("dataset_repeats_multiplier", 150),
                preset.get("dataset_repeats_multiplier", 150)
                * 100
                // dataset_info["total_images"],
            ),
        )

        config = {
            "client_id": client_id,
            "creation_date": datetime.now().isoformat(),
            "preset_name": preset["name"],
            "preset_key": preset["key"],
            "dataset_size": dataset_info["total_images"],
            "gpu_optimization": f"{gpu_profile.name.replace(' ', '_')}_{gpu_info.get('vram_gb', 0):.0f}GB",
            "detected_gpu": gpu_info.get("name", "Unknown"),
            "avatar_type": dataset_info.get("avatar_type", "unknown"),
            "model_config": {
                "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae": "madebyollin/sdxl-vae-fp16-fix",
                "v2": False,
                "v_parameterization": False,
                "clip_skip": 2,
            },
            "network_config": {
                "network_module": "networks.lora",
                "network_dim": gpu_profile.network_dim,
                "network_alpha": gpu_profile.network_alpha,
                "network_args": {
                    "conv_lora": gpu_profile.conv_lora,
                    "algo": "lora",
                },
            },
            "memory_optimizations": gpu_profile.memory_optimizations.copy(),
            "training_config": {
                "max_train_steps": preset["max_train_steps"],
                "learning_rate": preset["learning_rate"],
                "train_batch_size": gpu_profile.batch_size,
                "lr_scheduler": "cosine_with_restarts",
                "lr_warmup_steps": int(preset["max_train_steps"] * 0.1),
                "optimizer_type": gpu_profile.optimizer,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": gpu_profile.gradient_accumulation_steps,
            },
            "dataset_config": {
                "resolution": gpu_profile.resolution,
                "bucket_resolution_steps": 64,
                "bucket_no_upscale": True,
                "min_bucket_reso": 512,
                "max_bucket_reso": gpu_profile.resolution,
                "dataset_repeats": dataset_repeats,
                "shuffle_caption": True,
                "caption_extension": ".txt",
                "keep_tokens": 1,
            },
            "advanced_config": {
                "noise_offset": 0.1 if gpu_profile.conv_lora else 0.05,
                "adaptive_noise_scale": 0.05 if gpu_profile.conv_lora else None,
                "multires_noise_iterations": 4 if gpu_profile.conv_lora else 0,
                "multires_noise_discount": 0.3 if gpu_profile.conv_lora else 0,
                "ip_noise_gamma": None,
                "debiased_estimation_loss": gpu_profile.conv_lora,
            },
            "save_config": {
                "save_every_n_steps": preset.get("save_every_n_steps", 500),
                "save_model_as": "safetensors",
                "save_precision": "fp16",
                "output_name": f"{client_id}_avatar_lora_{preset['key']}",
                "max_checkpoints": 5 if gpu_profile.vram_gb_min >= 8 else 3,
            },
        }
        return config

    def _confirm_configuration(
        self, config: Dict, dataset_info: Dict, gpu_info: Dict
    ) -> bool:
        """Confirma configuración"""
        print(f"\n📋 RESUMEN DE CONFIGURACIÓN")
        print("=" * 50)
        print(f"Cliente: {config['client_id']}")
        print(f"Preset: {config['preset_name']}")
        print(f"Dataset: {dataset_info['total_images']} imágenes")
        print(f"GPU: {gpu_info.get('name', 'Unknown')}")

        training_config = config["training_config"]
        print(f"\n🎯 PARÁMETROS:")
        print(f"   Steps: {training_config['max_train_steps']:,}")
        print(f"   Learning rate: {training_config['learning_rate']}")
        print(f"   Batch size: {training_config['train_batch_size']}")

        if gpu_info.get("profile"):
            total_hours = (
                training_config["max_train_steps"]
                / gpu_info["profile"].steps_per_hour_estimate
            )
            print(f"\n⏱️ TIEMPO ESTIMADO: {total_hours:.1f} horas")

        return input("\n¿Guardar esta configuración? (s/n): ").lower().startswith("s")

    def _save_training_config(self, config: Dict, client_path: Path) -> bool:
        """Guarda configuración"""
        try:
            config_file = client_path / "training" / "lora_config.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            if save_json_safe(config, config_file, self.logger):
                print(f"✅ Configuración guardada")
                return True
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _setup_training_environment(self, client_path: Path) -> bool:
        """Configura entorno"""
        print(f"\n🔧 CONFIGURANDO ENTORNO")
        print("-" * 25)

        success = True
        if not self._setup_kohya_ss():
            success = False

        print(f"✅ Entorno configurado")
        input("Presiona Enter para continuar...")
        return success

    def _display_training_info(self, config: Dict, client_path: Path):
        """Muestra información de entrenamiento"""
        print(f"📋 INFORMACIÓN:")
        print(f"   Cliente: {config['client_id']}")
        print(f"   Preset: {config['preset_name']}")

        training_config = config["training_config"]
        print(f"   Steps: {training_config['max_train_steps']:,}")
        print(f"   Learning Rate: {training_config['learning_rate']}")

    def _confirm_training_start(self, config: Dict) -> bool:
        """Confirma inicio de entrenamiento"""
        training_config = config["training_config"]

        if self.detected_gpu_profile:
            estimated_hours = (
                training_config["max_train_steps"]
                / self.detected_gpu_profile.steps_per_hour_estimate
            )
        else:
            estimated_hours = training_config["max_train_steps"] / 200

        print(f"\n🚨 CONFIRMACIÓN FINAL")
        print(f"⏱️ Duración estimada: {estimated_hours:.1f} horas")
        print(f"🔥 Entrenamiento REAL - consumirá GPU durante horas")

        while True:
            confirm = input("\n¿Iniciar entrenamiento REAL? (si/no): ").strip().lower()
            if confirm in ["si", "s", "yes", "y"]:
                return True
            elif confirm in ["no", "n"]:
                return False
            else:
                print("Por favor responde 'si' o 'no'")

    def _update_training_history(
        self, client_path: Path, config: Dict, duration, success: bool
    ):
        """Actualiza historial"""
        try:
            config_file = client_path / "metadata" / "client_config.json"
            client_config = load_json_safe(config_file, {})

            if "training_history" not in client_config:
                client_config["training_history"] = []

            history_entry = {
                "date": datetime.now().isoformat(),
                "preset": config.get("preset_name", "Unknown"),
                "steps": config.get("training_config", {}).get("max_train_steps", 0),
                "success": success,
                "duration_seconds": duration.total_seconds() if duration else None,
                "gpu_used": config.get("detected_gpu", "Unknown"),
            }

            client_config["training_history"].append(history_entry)
            save_json_safe(client_config, config_file, self.logger)
        except Exception as e:
            self.logger.error(f"Error actualizando historial: {e}")

    # Métodos de gestión y progreso
    def show_training_progress(self, client_id: str, clients_dir: Path):
        """Muestra progreso del entrenamiento"""
        client_path = clients_dir / client_id
        print(f"\n📈 PROGRESO DE ENTRENAMIENTO - {client_id}")
        print("=" * 60)

        models_dir = client_path / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.safetensors"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                model_time = datetime.fromtimestamp(latest_model.stat().st_mtime)
                model_size = latest_model.stat().st_size / (1024 * 1024)
                print(f"🧠 ÚLTIMO MODELO:")
                print(f"   Archivo: {latest_model.name}")
                print(f"   Tamaño: {model_size:.1f}MB")
                print(f"   Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("📝 No hay modelos generados aún")
        else:
            print("📝 No hay entrenamiento iniciado")

        input("\nPresiona Enter para continuar...")

    def generate_test_samples(self, client_id: str, clients_dir: Path):
        """Gestiona muestras de prueba"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n🎨 GENERACIÓN DE MUESTRAS - {client_id}")
        print("=" * 50)

        if not models_dir.exists():
            print("❌ No se encontró directorio de modelos")
            input("Presiona Enter para continuar...")
            return

        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No hay modelos entrenados disponibles")
            input("Presiona Enter para continuar...")
            return

        print("🧠 MODELOS DISPONIBLES:")
        for i, model_file in enumerate(
            sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True), 1
        ):
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            model_size = model_file.stat().st_size / (1024 * 1024)
            print(f"   {i}. {model_file.name}")
            print(
                f"      Tamaño: {model_size:.1f}MB | Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        print(f"\n💡 PARA USAR EL MODELO:")
        print(f"   1. Trigger word: '{client_id}'")
        print(f"   2. Weight: 0.7-1.0")
        print(f"   3. Prompts ejemplo:")
        print(
            f"      - 'portrait of {client_id}, detailed face, professional lighting'"
        )
        print(f"      - '{client_id} smiling, high quality photography'")

        input("\nPresiona Enter para continuar...")

    def manage_trained_models(self, client_id: str, clients_dir: Path):
        """Gestiona modelos entrenados"""
        client_path = clients_dir / client_id
        models_dir = client_path / "models"

        print(f"\n📦 GESTIÓN DE MODELOS - {client_id}")
        print("=" * 45)

        if not models_dir.exists():
            print("❌ No se encontró directorio de modelos")
            input("Presiona Enter para continuar...")
            return

        model_files = list(models_dir.glob("*.safetensors"))
        if not model_files:
            print("❌ No hay modelos disponibles")
            input("Presiona Enter para continuar...")
            return

        total_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
        print(f"📊 ESTADÍSTICAS:")
        print(f"   Total modelos: {len(model_files)}")
        print(f"   Espacio usado: {total_size:.1f}MB")

        print(f"\n🧠 MODELOS DISPONIBLES:")
        for i, model_file in enumerate(
            sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True), 1
        ):
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
            model_size = model_file.stat().st_size / (1024 * 1024)

            if "-" in model_file.stem:
                parts = model_file.stem.split("-")
                step_count = parts[-1] if parts[-1].isdigit() else "final"
            else:
                step_count = "unknown"

            print(f"   {i}. {model_file.name}")
            print(f"      Steps: {step_count} | Tamaño: {model_size:.1f}MB")
            print(f"      Creado: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")

        input("\nPresiona Enter para continuar...")
