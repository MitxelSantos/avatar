#!/usr/bin/env python3
"""
Avatar Pipeline - Sistema Completo Profesional
Coordinador principal con arquitectura modular
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

        # Parámetros de calidad globales
        self.qc_params = {
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
        print("🎯 AVATAR PIPELINE - SISTEMA PROFESIONAL v2.0")
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

        # Preguntar si seleccionarlo
        if (
            input("\n¿Seleccionar este cliente para trabajar? (s/n): ")
            .lower()
            .startswith("s")
        ):
            self.current_client = client_id
            print(f"🎯 Cliente '{client_id}' seleccionado")

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

        # Usar face_processor para procesamiento completo
        success = self.face_processor.process_client_images(
            client_id=self.current_client,
            clients_dir=self.clients_dir,
            source_type=source_type,
            qc_params=self.qc_params,
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

        # Usar lora_trainer para configuración
        return self.lora_trainer.configure_training(
            client_id=self.current_client, clients_dir=self.clients_dir
        )

    def start_lora_training(self):
        if not self.current_client:
            print("❌ No hay cliente seleccionado")
            self.wait_input()
            return False

        # Usar lora_trainer para entrenamiento
        return self.lora_trainer.start_training(
            client_id=self.current_client, clients_dir=self.clients_dir
        )

    # === MENÚS ===

    def show_main_menu(self):
        self.clear_screen()
        self.show_header()

        print("\n🏠 MENÚ PRINCIPAL")
        print("1. 🏗️  Setup inicial del proyecto")
        print("2. 👥 Gestión de clientes")
        print("3. 📊 Ver estadísticas generales")
        print("4. ⚙️  Configurar parámetros globales")
        print("5. 🚪 Salir")

        return input("\nSelecciona una opción (1-5): ").strip()

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
            print("4. ⚙️  Configurar entrenamiento LoRA")
            print("5. 🚀 Iniciar entrenamiento")
            print("6. 📈 Ver progreso de entrenamiento")
            if has_models:
                print("7. 🎨 Generar muestras de prueba")
                print("8. 📦 Gestionar modelos entrenados")
        else:
            print("4. ❌ Entrenamiento LoRA (necesita dataset procesado)")

        print(f"\n🔧 CONFIGURACIÓN:")
        print("9. 📊 Ver estado del cliente")
        print("10. 🔧 Configurar parámetros específicos")
        print("11. 🔙 Cambiar de cliente")
        print("12. 🏠 Volver al menú principal")

        max_option = 12
        return input(f"\nSelecciona una opción (1-{max_option}): ").strip()

    # === FUNCIONES DE CONTROL ===

    def run_main(self):
        print("🚀 Iniciando Avatar Pipeline - Sistema Profesional")
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
            elif choice == "5":
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
                self.create_client()
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
            elif choice == "11":
                self.select_client()
            elif choice == "12":
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

        print("Configuración actual:")
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
            # Restaurar defaults
            self.qc_params = {
                "face_confidence_threshold": 0.85,
                "face_padding_factor": 1.6,
                "min_file_size_kb": 200,
                "max_file_size_mb": 5,
                "min_brightness": 40,
                "max_brightness": 220,
                "min_contrast": 25,
                "blur_threshold": 100,
            }
            print("✅ Parámetros restaurados")
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
