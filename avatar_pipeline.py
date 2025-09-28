#!/usr/bin/env python3
"""
Avatar Pipeline - VERSIÓN CON FIXES DE NAVEGACIÓN Y CUDA
FIXES APLICADOS:
- Navegación correcta al crear cliente
- Mejor detección e instrucciones para CUDA
- Validación de PyTorch con CUDA específica
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
import warnings

# Suprimir warnings molestos
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# Imports de módulos especializados
try:
    from image_processor import FaceProcessor
    from data_preprocessor import DataPreprocessor
    from lora_trainer import LoRATrainer
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que todos los archivos estén en el mismo directorio")
    sys.exit(1)


class AvatarPipeline:
    def __init__(self):
        self.base_dir = Path(".")
        self.clients_dir = self.base_dir / "clients"
        self.training_dir = self.base_dir / "training"
        self.current_client = None

        # Inicializar módulos especializados
        self.face_processor = FaceProcessor()
        self.data_preprocessor = DataPreprocessor()
        self.lora_trainer = LoRATrainer()

        # PARÁMETROS DE CALIDAD GLOBALES - OPTIMIZADOS PARA RAW
        self.qc_params = {
            "face_confidence_threshold": 0.85,
            "face_padding_factor": 1.6,
            "min_file_size_kb": 200,
            "max_file_size_mb": 5,
            "min_brightness": 30,  # Bajado de 40 para RAW
            "max_brightness": 240,  # Subido de 220 para RAW
            "min_contrast": 15,  # Bajado de 25 para RAW
            "blur_threshold": 10,  # Bajado SIGNIFICATIVAMENTE de 100 para RAW
        }

    def get_qc_params_for_source(self, is_raw_source=False):
        """Devuelve parámetros de QC optimizados según el tipo de fuente"""
        if is_raw_source:
            # Parámetros más tolerantes para archivos RAW convertidos
            return {
                "face_confidence_threshold": 0.85,
                "face_padding_factor": 1.6,
                "min_file_size_kb": 200,
                "max_file_size_mb": 8,  # RAW puede generar archivos más grandes
                "min_brightness": 25,  # Más tolerante
                "max_brightness": 250,  # Más tolerante
                "min_contrast": 12,  # Más tolerante - RAW puede tener menos contraste aparente
                "blur_threshold": 8,  # MUY tolerante - RAW convertido puede parecer menos nítido
            }
        else:
            # Parámetros estrictos para archivos nativos (JPEG, PNG)
            return {
                "face_confidence_threshold": 0.85,
                "face_padding_factor": 1.6,
                "min_file_size_kb": 200,
                "max_file_size_mb": 5,
                "min_brightness": 40,
                "max_brightness": 220,
                "min_contrast": 25,
                "blur_threshold": 100,
            }

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_header(self):
        print("=" * 70)
        print("🎯 AVATAR PIPELINE")
        print("=" * 70)
        if self.current_client:
            print(f"📋 Cliente Actual: {self.current_client}")
            print("-" * 70)

    def wait_input(self, msg="Presiona Enter para continuar..."):
        input(f"\n{msg}")

    def get_directory_dialog(self, title="Seleccionar Directorio"):
        try:
            import tkinter as tk
            from tkinter import filedialog

            print(f"📁 {title}")
            print("Se abrirá una ventana para seleccionar el directorio...")

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            directory = filedialog.askdirectory(title=title)
            root.destroy()

            return directory if directory else None

        except ImportError:
            print("⚠️ tkinter no disponible. Ingresa la ruta manualmente:")
            return input("Ruta del directorio: ").strip()

    def check_cuda_installation(self):
        """Verifica instalación de CUDA con diagnóstico detallado"""
        print("\n🔍 DIAGNÓSTICO DE CUDA")
        print("-" * 25)

        try:
            import torch

            print(f"✅ PyTorch {torch.__version__} instalado")

            # Verificar CUDA disponible
            cuda_available = torch.cuda.is_available()
            print(f"CUDA disponible: {'✅' if cuda_available else '❌'}")

            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"Dispositivos CUDA: {device_count}")

                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (
                        1024**3
                    )
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

                # Verificar versión CUDA
                cuda_version = torch.version.cuda
                print(f"Versión CUDA: {cuda_version}")

                return True
            else:
                print("\n❌ CUDA NO DISPONIBLE")
                print("Posibles causas:")
                print("1. PyTorch instalado sin soporte CUDA")
                print("2. Drivers NVIDIA no instalados o desactualizados")
                print("3. CUDA Toolkit no instalado")

                return False

        except ImportError:
            print("❌ PyTorch no instalado")
            return False

    def fix_cuda_installation(self):
        """Proporciona instrucciones específicas para corregir CUDA"""
        print("\n🔧 SOLUCIONES PARA CUDA")
        print("=" * 25)

        print("PASO 1: Verificar GPU NVIDIA")
        print("  - Abre el Administrador de dispositivos")
        print("  - Busca en 'Adaptadores de pantalla'")
        print("  - Debe aparecer una GPU NVIDIA")

        print("\nPASO 2: Instalar/Actualizar Drivers NVIDIA")
        print("  - Visita: https://www.nvidia.com/drivers/")
        print("  - Descarga el driver más reciente para tu GPU")
        print("  - Reinicia después de la instalación")

        print("\nPASO 3: Reinstalar PyTorch con CUDA")
        print("  Desinstalar PyTorch actual:")
        print("    pip uninstall torch torchvision torchaudio")
        print("")
        print("  Instalar PyTorch con CUDA 11.8:")
        print(
            "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        print("")
        print("  O con CUDA 12.1:")
        print(
            "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )

        print("\nPASO 4: Verificar instalación")
        print(
            "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
        )

        return input("\n¿Continuar sin CUDA? (s/n): ").lower().startswith("s")

    # === GESTIÓN DE CLIENTES ===

    def setup_project(self):
        self.clear_screen()
        self.show_header()

        print("\n🏗️ SETUP INICIAL DEL PROYECTO")
        print("-" * 40)

        # Crear directorios
        for dir_path in [self.clients_dir, self.training_dir]:
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Creado: {dir_path}")

        # Mover lora-clients si existe
        old_clients = self.base_dir / "lora-clients"
        if old_clients.exists():
            print("\n📁 Moviendo lora-clients a clients...")
            for item in old_clients.iterdir():
                dst = self.clients_dir / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
                    print(f"  ✅ Movido: {item.name}")
            shutil.rmtree(old_clients)

        print("\n✅ Setup completado!")
        print("📋 Pasos siguientes:")
        print("   1. Crear un cliente nuevo")
        print("   2. Cargar imágenes MJ con metadata completa")
        print("   3. Procesar para entrenamiento LoRA de máxima calidad")

        self.wait_input()

    def create_client(self):
        """VERSIÓN CORREGIDA: Navega automáticamente al menú de operaciones"""
        self.clear_screen()
        self.show_header()

        print("\n➕ CREAR NUEVO CLIENTE")
        print("-" * 30)

        while True:
            client_id = input("\nNombre del cliente: ").strip()

            if not client_id:
                print("❌ El nombre no puede estar vacío")
                continue

            client_path = self.clients_dir / client_id
            if client_path.exists():
                print(f"❌ El cliente '{client_id}' ya existe")
                continue

            break

        # Crear estructura completa
        subdirs = [
            "raw_mj",  # Imágenes MJ originales
            "raw_real",  # Fotos reales originales
            "processed",  # Imágenes procesadas 1024x1024
            "dataset_lora",  # Dataset final para entrenamiento
            "rejected",  # Imágenes rechazadas por QC
            "metadata",  # Logs, CSVs, configuraciones
            "training",  # Checkpoints y logs de entrenamiento
            "models",  # Modelos LoRA finales
            "samples",  # Muestras generadas durante entrenamiento
            "output",  # Exports y resultados finales
        ]

        print(f"\n🏗️ Creando estructura para: {client_id}")
        for subdir in subdirs:
            (client_path / subdir).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {subdir}/")

        # Crear archivo de configuración del cliente
        client_config = {
            "client_id": client_id,
            "created_date": datetime.now().isoformat(),
            "omni_weight": 160,  # Default, puede modificarse
            "prompt_maestro": "",  # Se llenará al importar MJ
            "processing_settings": self.qc_params.copy(),
            "training_history": [],
            "status": "created",
        }

        config_file = client_path / "metadata" / "client_config.json"
        with open(config_file, "w") as f:
            json.dump(client_config, f, indent=2)

        print(f"\n✅ Cliente '{client_id}' creado exitosamente!")
        print(f"📋 Configuración guardada en: {config_file}")

        # FIX: Preguntar si seleccionarlo Y navegar automáticamente
        select_client = (
            input("\n¿Seleccionar este cliente para trabajar? (s/n): ").lower().strip()
        )

        if select_client.startswith("s"):
            self.current_client = client_id
            print(f"🎯 Cliente '{client_id}' seleccionado")
            print(f"🚀 Navegando al menú de operaciones del cliente...")

            time.sleep(1)  # Pausa breve para que el usuario vea el mensaje

            # CORRECCIÓN: Ir directamente al menú de operaciones
            self.run_client_operations()
            return "operations"  # Señal especial para salir del bucle de gestión
        else:
            self.wait_input()
            return True

    def list_clients(self):
        self.clear_screen()
        self.show_header()

        print("\n📋 CLIENTES EXISTENTES")
        print("-" * 40)

        if not self.clients_dir.exists():
            print("❌ No se encontró el directorio de clientes")
            print("💡 Ejecuta 'Setup inicial del proyecto' primero")
            self.wait_input()
            return

        # Obtener lista de clientes
        client_list = []
        try:
            for item in self.clients_dir.iterdir():
                if item.is_dir():
                    client_list.append(item.name)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.wait_input()
            return

        if not client_list:
            print("📁 No hay clientes creados")
            print("💡 Usa 'Crear nuevo cliente' para empezar")
            self.wait_input()
            return

        print(f"Encontrados {len(client_list)} clientes:\n")

        for i, client in enumerate(sorted(client_list), 1):
            client_path = self.clients_dir / client

            # Contar archivos
            mj_count = (
                len(list((client_path / "raw_mj").glob("*")))
                if (client_path / "raw_mj").exists()
                else 0
            )
            real_count = (
                len(list((client_path / "raw_real").glob("*")))
                if (client_path / "raw_real").exists()
                else 0
            )
            lora_count = (
                len(list((client_path / "dataset_lora").glob("*")))
                if (client_path / "dataset_lora").exists()
                else 0
            )
            models_count = (
                len(list((client_path / "models").glob("*.safetensors")))
                if (client_path / "models").exists()
                else 0
            )

            # Determinar status
            if models_count > 0:
                status = "🎯 Modelo Listo"
            elif lora_count > 0:
                status = "🚀 Preparado para LoRA"
            elif mj_count > 0 or real_count > 0:
                status = "🔄 Procesando"
            else:
                status = "📝 Nuevo"

            print(f"{i:2d}. {client}")
            print(
                f"    MJ: {mj_count:3d} | Real: {real_count:3d} | Dataset: {lora_count:3d} | Modelos: {models_count:1d}"
            )
            print(f"    Status: {status}")
            print()

        self.wait_input()

    def select_client(self):
        self.clear_screen()
        self.show_header()

        print("\n🎯 SELECCIONAR CLIENTE")
        print("-" * 25)

        # Verificar directorio
        if not self.clients_dir.exists():
            print("❌ No hay directorio de clientes. Ejecuta setup primero.")
            self.wait_input()
            return False

        # Obtener clientes
        client_list = []
        for item in self.clients_dir.iterdir():
            if item.is_dir():
                client_list.append(item.name)

        if not client_list:
            print("❌ No hay clientes disponibles. Crea uno primero.")
            self.wait_input()
            return False

        # Mostrar opciones
        sorted_clients = sorted(client_list)
        for i, client in enumerate(sorted_clients, 1):
            current_mark = " 👈 ACTUAL" if client == self.current_client else ""
            print(f"{i:2d}. {client}{current_mark}")

        print(f"{len(sorted_clients) + 1:2d}. 🔙 Cancelar")

        # Obtener selección
        while True:
            try:
                choice = int(
                    input(f"\nSelecciona cliente (1-{len(sorted_clients) + 1}): ")
                )

                if choice == len(sorted_clients) + 1:
                    return False

                if 1 <= choice <= len(sorted_clients):
                    self.current_client = sorted_clients[choice - 1]
                    print(f"\n🎯 Cliente '{self.current_client}' seleccionado")
                    self.wait_input()
                    return True
                else:
                    print("❌ Opción inválida")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                print("\n🔙 Cancelado")
                return False

    def show_client_status(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return

        self.clear_screen()
        self.show_header()

        client_path = self.clients_dir / self.current_client

        print(f"\n📊 ESTADO DETALLADO: {self.current_client}")
        print("-" * 50)
        print(f"📁 Ruta: {client_path}")

        # Cargar configuración del cliente
        config_file = client_path / "metadata" / "client_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

            print(f"\n⚙️ CONFIGURACIÓN:")
            print(f"   Creado: {config.get('created_date', 'N/A')}")
            print(f"   Omni Weight: {config.get('omni_weight', 'N/A')}")
            print(
                f"   Prompt Maestro: {config.get('prompt_maestro', 'No configurado')[:50]}..."
            )
            print(f"   Entrenamientos: {len(config.get('training_history', []))}")

        # Contar archivos por directorio
        subdirs = [
            "raw_mj",
            "raw_real",
            "processed",
            "dataset_lora",
            "rejected",
            "models",
            "samples",
        ]
        total_files = 0

        print(f"\n📂 CONTENIDO POR DIRECTORIO:")
        for subdir in subdirs:
            subdir_path = client_path / subdir
            if subdir_path.exists():
                if subdir == "models":
                    count = len(list(subdir_path.glob("*.safetensors")))
                else:
                    count = len(list(subdir_path.glob("*")))
                total_files += count

                # Icons por tipo
                icons = {
                    "raw_mj": "🎨",
                    "raw_real": "📷",
                    "processed": "🔄",
                    "dataset_lora": "🎯",
                    "rejected": "❌",
                    "models": "🧠",
                    "samples": "🖼️",
                }
                icon = icons.get(subdir, "📂")
                print(f"   {icon} {subdir:15s}: {count:3d} archivos")

        print(f"\n📊 Total de archivos: {total_files}")

        # Status de entrenamiento
        if (client_path / "training").exists():
            training_logs = list((client_path / "training").glob("*.log"))
            if training_logs:
                latest_log = max(training_logs, key=lambda x: x.stat().st_mtime)
                print(f"📈 Último entrenamiento: {latest_log.name}")

        self.wait_input()

    # === OPERACIONES PRINCIPALES ===

    def load_mj_images(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        self.clear_screen()
        self.show_header()

        print(f"\n🎨 CARGAR IMÁGENES MIDJOURNEY")
        print(f"Cliente: {self.current_client}")
        print("-" * 40)

        # Seleccionar directorio
        source_dir = self.get_directory_dialog(
            "Seleccionar directorio de imágenes MidJourney"
        )

        if not source_dir:
            print("❌ No se seleccionó directorio")
            self.wait_input()
            return False

        print(f"\n📁 Directorio seleccionado: {source_dir}")

        # Usar data_preprocessor para procesar imágenes MJ con metadata completa
        success = self.data_preprocessor.process_mj_images(
            client_id=self.current_client,
            source_dir=source_dir,
            clients_dir=self.clients_dir,
        )

        if success:
            print(f"\n✅ Imágenes MJ importadas con metadata completa")

            # Preguntar si procesar inmediatamente
            if (
                input("\n¿Procesar imágenes con detección facial ahora? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self.process_images("mj")

        self.wait_input()
        return success

    def load_real_images(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        self.clear_screen()
        self.show_header()

        print(f"\n📷 CARGAR FOTOS REALES")
        print(f"Cliente: {self.current_client}")
        print("-" * 40)

        # Seleccionar directorio
        source_dir = self.get_directory_dialog("Seleccionar directorio de fotos reales")

        if not source_dir:
            print("❌ No se seleccionó directorio")
            self.wait_input()
            return False

        print(f"\n📁 Directorio seleccionado: {source_dir}")

        # Usar data_preprocessor para procesar fotos reales
        success = self.data_preprocessor.process_real_images(
            client_id=self.current_client,
            source_dir=source_dir,
            clients_dir=self.clients_dir,
        )

        if success:
            print(f"\n✅ Fotos reales importadas y analizadas")

            # Preguntar si procesar inmediatamente
            if (
                input("\n¿Procesar fotos con detección facial ahora? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self.process_images("real")

        self.wait_input()
        return success

    def process_images(self, source_type="all"):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        self.clear_screen()
        self.show_header()

        print(f"\n🔄 PROCESAMIENTO FACIAL AVANZADO - {source_type.upper()}")
        print(f"Cliente: {self.current_client}")
        print("-" * 50)

        # DETECTAR SI HAY ARCHIVOS RAW PARA USAR PARÁMETROS ADAPTATIVOS
        client_path = self.clients_dir / self.current_client
        has_raw_files = False

        if source_type in ["all", "real"]:
            raw_real_dir = client_path / "raw_real"
            if raw_real_dir.exists():
                raw_extensions = [".nef", ".cr2", ".arw", ".dng", ".raf", ".orf"]
                for file_path in raw_real_dir.iterdir():
                    if file_path.suffix.lower() in raw_extensions:
                        has_raw_files = True
                        break

        # Usar parámetros adaptativos
        if has_raw_files:
            qc_params = self.get_qc_params_for_source(is_raw_source=True)
            print("🔧 Usando parámetros optimizados para archivos RAW")
            print("   📸 Blur threshold: 8 (muy tolerante para RAW)")
            print("   🔍 Contrast threshold: 12 (tolerante para RAW)")
        else:
            qc_params = self.get_qc_params_for_source(is_raw_source=False)
            print("🔧 Usando parámetros estándar para archivos nativos")

        # CORRECCIÓN: Usar force_qc_params en lugar de qc_params
        success = self.face_processor.process_client_images(
            client_id=self.current_client,
            clients_dir=self.clients_dir,
            source_type=source_type,
            force_qc_params=qc_params,  # ← CORREGIDO: force_qc_params
        )

        if success:
            print(f"\n🎉 Procesamiento completado exitosamente!")

            # Preguntar si preparar para entrenamiento LoRA
            if (
                input("\n¿Preparar automáticamente para entrenamiento LoRA? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self.prepare_lora_training()

        self.wait_input()
        return success

    def prepare_lora_training(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        print(f"\n🎯 PREPARANDO ENTRENAMIENTO LORA...")

        # Usar data_preprocessor para generar captions y balance de datos
        success = self.data_preprocessor.prepare_lora_dataset(
            client_id=self.current_client, clients_dir=self.clients_dir
        )

        if success:
            print(f"✅ Dataset LoRA preparado con balance 85% MJ / 15% Real")

            # Preguntar si configurar entrenamiento
            if (
                input("\n¿Configurar parámetros de entrenamiento ahora? (s/n): ")
                .lower()
                .startswith("s")
            ):
                return self.configure_lora_training()

        return success

    def configure_lora_training(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        print(f"\n⚙️ CONFIGURANDO ENTRENAMIENTO LORA...")
        print(f"Cliente: {self.current_client}")
        print("-" * 40)

        try:
            # Usar lora_trainer para configuración
            result = self.lora_trainer.configure_training(
                client_id=self.current_client, clients_dir=self.clients_dir
            )

            if result:
                print(f"\n✅ Configuración de entrenamiento completada")
                print(f"📋 El entrenamiento está listo para iniciarse")
            else:
                print(f"\n❌ Configuración cancelada o falló")

            self.wait_input()
            return result

        except Exception as e:
            print(f"\n❌ Error en configuración: {str(e)}")
            print(f"🔧 Detalles del error para debug:")
            import traceback

            traceback.print_exc()
            self.wait_input()
            return False

    def start_lora_training(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        # MEJORA: Verificar CUDA antes de iniciar entrenamiento
        print(f"\n🚀 VERIFICANDO PREREQUISITOS PARA ENTRENAMIENTO")
        print("=" * 55)

        # Diagnóstico CUDA detallado
        cuda_ok = self.check_cuda_installation()

        if not cuda_ok:
            print(f"\n⚠️ ADVERTENCIA: CUDA NO DISPONIBLE")
            print("El entrenamiento LoRA requiere GPU con CUDA para ser eficiente.")
            print(
                "Sin CUDA, el entrenamiento será extremadamente lento (días vs horas)."
            )

            continue_anyway = (
                input("\n¿Continuar sin CUDA? (NO recomendado) (s/n): ").lower().strip()
            )
            if not continue_anyway.startswith("s"):
                # Mostrar instrucciones para corregir CUDA
                if (
                    input("\n¿Ver instrucciones para instalar CUDA? (s/n): ")
                    .lower()
                    .startswith("s")
                ):
                    self.fix_cuda_installation()
                return False

        print(f"\n🚀 INICIANDO ENTRENAMIENTO LORA...")
        print(f"Cliente: {self.current_client}")
        print("-" * 40)

        try:
            # Usar lora_trainer para entrenamiento con debug mejorado
            print(f"🔍 Verificando prerequisitos...")

            client_path = self.clients_dir / self.current_client

            # Verificaciones previas con debug
            dataset_dir = client_path / "dataset_lora"
            config_file = client_path / "training" / "lora_config.json"

            print(f"📁 Dataset dir existe: {dataset_dir.exists()}")
            print(f"📄 Config file existe: {config_file.exists()}")

            if dataset_dir.exists():
                dataset_images = list(dataset_dir.glob("*.png"))
                print(f"🖼️ Imágenes en dataset: {len(dataset_images)}")

            # Ejecutar entrenamiento
            result = self.lora_trainer.start_training(
                client_id=self.current_client, clients_dir=self.clients_dir
            )

            if result:
                print(f"\n🎉 ¡Entrenamiento completado exitosamente!")
                print(f"🧠 Modelo entrenado disponible en:")
                print(f"   {client_path / 'models'}")
            else:
                print(f"\n❌ Entrenamiento falló o fue cancelado")
                print(f"📋 Revisa los logs para más detalles")

            self.wait_input()
            return result

        except Exception as e:
            print(f"\n❌ Error inesperado en entrenamiento: {str(e)}")
            print(f"🔧 Detalles del error para debug:")
            import traceback

            traceback.print_exc()
            self.wait_input()
            return False

    # === MENÚS ===

    def show_main_menu(self):
        self.clear_screen()
        self.show_header()

        print("\n🏠 MENÚ PRINCIPAL")
        print("1. 🏗️  Setup inicial del proyecto")
        print("2. 👥 Gestión de clientes")
        print("3. 📊 Ver estadísticas generales")
        print("4. ⚙️  Configurar parámetros globales")
        print("5. 🔍 Diagnóstico de CUDA")  # NUEVA OPCIÓN
        print("6. 🚪 Salir")

        return input("\nSelecciona una opción (1-6): ").strip()

    def show_client_menu(self):
        self.clear_screen()
        self.show_header()

        print("\n👥 GESTIÓN DE CLIENTES")
        print("1. 📋 Listar clientes existentes")
        print("2. ➕ Crear nuevo cliente")
        print("3. 🎯 Seleccionar cliente para trabajar")
        print("4. 📊 Ver estado detallado de cliente")
        print("5. 🔙 Volver al menú principal")

        return input("\nSelecciona una opción (1-5): ").strip()

    def show_client_operations_menu(self):
        self.clear_screen()
        self.show_header()

        # Verificar estado del cliente
        client_path = self.clients_dir / self.current_client
        dataset_ready = (client_path / "dataset_lora").exists() and len(
            list((client_path / "dataset_lora").glob("*.png"))
        ) > 0
        config_ready = (client_path / "training" / "lora_config.json").exists()
        has_models = (client_path / "models").exists() and len(
            list((client_path / "models").glob("*.safetensors"))
        ) > 0

        print(f"\n🎯 OPERACIONES - {self.current_client}")
        print("=" * 50)

        print("\n📥 IMPORTACIÓN DE DATOS:")
        print("1. 🎨 Cargar imágenes MidJourney")
        print("2. 📷 Cargar fotos reales")
        print("3. 🔄 Procesar imágenes cargadas")

        print(f"\n🧠 ENTRENAMIENTO LoRA:")
        if dataset_ready:
            status_4 = "✅" if config_ready else "⚙️"
            status_5 = "🚀" if config_ready else "❌"

            print(f"4. {status_4} Configurar entrenamiento LoRA")
            print(f"5. {status_5} Iniciar entrenamiento")
            print("6. 📈 Ver progreso de entrenamiento")

            if has_models:
                print("7. 🎨 Generar muestras de prueba")
                print("8. 📦 Gestionar modelos entrenados")
        else:
            print("4. ❌ Entrenamiento LoRA (necesita dataset procesado)")
            print("5. ❌ Entrenamiento LoRA (necesita dataset procesado)")

        print(f"\n🔧 CONFIGURACIÓN:")
        print("9. 📊 Ver estado del cliente")
        print("10. 🔧 Configurar parámetros específicos")
        print("11. 🔍 Diagnóstico de CUDA")  # NUEVA OPCIÓN
        print("12. 🔙 Cambiar de cliente")
        print("13. 🏠 Volver al menú principal")

        # Mostrar hints sobre el estado
        print(f"\n💡 ESTADO ACTUAL:")
        print(f"   Dataset preparado: {'✅' if dataset_ready else '❌'}")
        print(f"   Configuración LoRA: {'✅' if config_ready else '❌'}")
        print(f"   Modelos entrenados: {'✅' if has_models else '❌'}")

        max_option = 13
        return input(f"\nSelecciona una opción (1-{max_option}): ").strip()

    # === FUNCIONES DE CONTROL ===

    def run_main(self):
        print("🚀 Iniciando Avatar Pipeline - VERSIÓN CON FIXES")
        print("✅ Navegación corregida + Diagnóstico CUDA mejorado")
        time.sleep(1)

        while True:
            choice = self.show_main_menu()

            if choice == "1":
                self.setup_project()
            elif choice == "2":
                self.run_client_management()
            elif choice == "3":
                self.show_general_stats()
            elif choice == "4":
                self.configure_global_params()
            elif choice == "5":  # NUEVA OPCIÓN
                self.check_cuda_installation()
                if not self.check_cuda_installation():
                    self.fix_cuda_installation()
                self.wait_input()
            elif choice == "6":
                self.clear_screen()
                print("\n👋 Sistema cerrado correctamente!")
                break
            else:
                print("❌ Opción inválida")
                time.sleep(1)

    def run_client_management(self):
        while True:
            choice = self.show_client_menu()

            if choice == "1":
                self.list_clients()
            elif choice == "2":
                # FIX: Manejar el caso especial de navegación directa
                result = self.create_client()
                if result == "operations":
                    # Si create_client retorna "operations", salir del bucle
                    break
            elif choice == "3":
                if self.select_client():
                    self.run_client_operations()
            elif choice == "4":
                self.show_client_status()
            elif choice == "5":
                break
            else:
                print("❌ Opción inválida")
                time.sleep(1)

    def run_client_operations(self):
        while True:
            choice = self.show_client_operations_menu()

            if choice == "1":
                self.load_mj_images()
            elif choice == "2":
                self.load_real_images()
            elif choice == "3":
                self.process_images()
            elif choice == "4":
                self.configure_lora_training()
            elif choice == "5":
                self.start_lora_training()
            elif choice == "6":
                self.lora_trainer.show_training_progress(
                    self.current_client, self.clients_dir
                )
            elif choice == "7":
                self.lora_trainer.generate_test_samples(
                    self.current_client, self.clients_dir
                )
            elif choice == "8":
                self.lora_trainer.manage_trained_models(
                    self.current_client, self.clients_dir
                )
            elif choice == "9":
                self.show_client_status()
            elif choice == "10":
                self.configure_client_params()
            elif choice == "11":  # NUEVA OPCIÓN
                self.check_cuda_installation()
                if not self.check_cuda_installation():
                    self.fix_cuda_installation()
                self.wait_input()
            elif choice == "12":
                self.select_client()
            elif choice == "13":
                break
            else:
                print("❌ Opción inválida")
                time.sleep(1)

    def show_general_stats(self):
        # Implementación de estadísticas generales
        self.clear_screen()
        self.show_header()

        print("\n📊 ESTADÍSTICAS GENERALES DEL SISTEMA")
        print("-" * 50)

        if not self.clients_dir.exists():
            print("❌ No se encontró directorio de clientes")
            self.wait_input()
            return

        # Obtener todos los clientes
        client_list = []
        for item in self.clients_dir.iterdir():
            if item.is_dir():
                client_list.append(item.name)

        if not client_list:
            print("📁 No hay clientes creados")
            self.wait_input()
            return

        # Estadísticas agregadas
        total_clients = len(client_list)
        total_mj = 0
        total_real = 0
        total_processed = 0
        total_models = 0
        clients_ready = 0

        print(f"👥 Total de clientes: {total_clients}\n")

        for client in client_list:
            client_path = self.clients_dir / client

            mj_count = (
                len(list((client_path / "raw_mj").glob("*")))
                if (client_path / "raw_mj").exists()
                else 0
            )
            real_count = (
                len(list((client_path / "raw_real").glob("*")))
                if (client_path / "raw_real").exists()
                else 0
            )
            processed_count = (
                len(list((client_path / "dataset_lora").glob("*")))
                if (client_path / "dataset_lora").exists()
                else 0
            )
            models_count = (
                len(list((client_path / "models").glob("*.safetensors")))
                if (client_path / "models").exists()
                else 0
            )

            total_mj += mj_count
            total_real += real_count
            total_processed += processed_count
            total_models += models_count

            if models_count > 0:
                clients_ready += 1

        print(f"📊 DATOS TOTALES:")
        print(f"   🎨 Imágenes MJ: {total_mj:,}")
        print(f"   📷 Fotos reales: {total_real:,}")
        print(f"   🎯 Imágenes procesadas: {total_processed:,}")
        print(f"   🧠 Modelos entrenados: {total_models}")
        print(f"   ✅ Clientes completados: {clients_ready}/{total_clients}")

        if total_clients > 0:
            print(f"\n📈 PROMEDIOS POR CLIENTE:")
            print(f"   MJ: {total_mj/total_clients:.1f}")
            print(f"   Real: {total_real/total_clients:.1f}")
            print(f"   Procesadas: {total_processed/total_clients:.1f}")

        self.wait_input()

    def configure_global_params(self):
        # Configuración de parámetros globales
        self.clear_screen()
        self.show_header()

        print("\n⚙️ CONFIGURACIÓN GLOBAL DE CALIDAD")
        print("-" * 40)

        print("Configuración actual (optimizada para RAW):")
        for key, value in self.qc_params.items():
            print(f"   {key}: {value}")

        print("\nOpciones:")
        print("1. Modificar parámetro específico")
        print("2. Restaurar valores por defecto")
        print("3. Volver")

        choice = input("\nSelecciona opción: ").strip()

        if choice == "1":
            # Implementar modificación de parámetros
            pass
        elif choice == "2":
            # Restaurar defaults optimizados para RAW
            self.qc_params = {
                "face_confidence_threshold": 0.85,
                "face_padding_factor": 1.6,
                "min_file_size_kb": 200,
                "max_file_size_mb": 5,
                "min_brightness": 30,
                "max_brightness": 240,
                "min_contrast": 15,
                "blur_threshold": 10,
            }
            print("✅ Parámetros restaurados (optimizados para RAW)")
            self.wait_input()

    def configure_client_params(self):
        # Configuración específica del cliente actual
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return

        self.clear_screen()
        self.show_header()

        print(f"\n🔧 CONFIGURACIÓN - {self.current_client}")
        print("-" * 40)

        client_path = self.clients_dir / self.current_client
        config_file = client_path / "metadata" / "client_config.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

            print("Configuración actual:")
            print(f"   Omni Weight: {config.get('omni_weight', 160)}")
            print(
                f"   Prompt Maestro: {config.get('prompt_maestro', 'No configurado')[:50]}..."
            )

            print("\nOpciones:")
            print("1. Modificar Omni Weight")
            print("2. Actualizar Prompt Maestro")
            print("3. Ver historial de entrenamientos")
            print("4. Volver")

            choice = input("\nSelecciona opción: ").strip()

            # Implementar modificaciones según choice

        self.wait_input()


if __name__ == "__main__":
    pipeline = AvatarPipeline()
    pipeline.run_main()
