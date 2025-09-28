#!/usr/bin/env python3
"""
utils.py - Utilidades comunes y sistema de logging para Avatar Pipeline
Versión 3.0 - Centralizado y robusto
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import shutil


class PipelineLogger:
    """Sistema de logging centralizado para Avatar Pipeline"""

    def __init__(self, name: str, log_dir: Path, level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configurar logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Limpiar handlers existentes
        self.logger.handlers.clear()

        # Handler para archivo
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def critical(self, message: str):
        self.logger.critical(message)


class ProgressTracker:
    """Tracker de progreso para operaciones largas"""

    def __init__(self, total: int, description: str = "Procesando"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update = 0

    def update(self, increment: int = 1, status: str = ""):
        self.current += increment

        # Actualizar cada 5% o cada 5 segundos
        now = time.time()
        progress_percent = (self.current / self.total) * 100

        if now - self.last_update >= 5 or progress_percent % 5 < 1:
            elapsed = now - self.start_time
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = format_time(eta)
            else:
                eta_str = "N/A"

            status_msg = f" - {status}" if status else ""
            print(
                f"📊 {self.description}: {self.current}/{self.total} ({progress_percent:.1f}%) | ETA: {eta_str}{status_msg}"
            )
            self.last_update = now

    def finish(self, message: str = "Completado"):
        elapsed = time.time() - self.start_time
        print(f"✅ {message} - Total: {format_time(elapsed)}")


def format_time(seconds: float) -> str:
    """Formatea tiempo en formato legible"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_size: int) -> str:
    """Formatea tamaño de archivo en formato legible"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def safe_create_directory(path: Path, logger: Optional[PipelineLogger] = None) -> bool:
    """Crea directorio de forma segura"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.debug(f"Directorio creado: {path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error creando directorio {path}: {e}")
        return False


def safe_copy_file(
    src: Path, dst: Path, logger: Optional[PipelineLogger] = None
) -> bool:
    """Copia archivo de forma segura"""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if logger:
            logger.debug(f"Archivo copiado: {src} -> {dst}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error copiando archivo {src} -> {dst}: {e}")
        return False


def safe_move_file(
    src: Path, dst: Path, logger: Optional[PipelineLogger] = None
) -> bool:
    """Mueve archivo de forma segura"""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        if logger:
            logger.debug(f"Archivo movido: {src} -> {dst}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error moviendo archivo {src} -> {dst}: {e}")
        return False


def safe_delete_file(path: Path, logger: Optional[PipelineLogger] = None) -> bool:
    """Elimina archivo de forma segura"""
    try:
        if path.exists():
            path.unlink()
            if logger:
                logger.debug(f"Archivo eliminado: {path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error eliminando archivo {path}: {e}")
        return False


def load_json_safe(
    path: Path, default: Any = None, logger: Optional[PipelineLogger] = None
) -> Any:
    """Carga JSON de forma segura"""
    try:
        if not path.exists():
            return default

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if logger:
            logger.debug(f"JSON cargado: {path}")
        return data

    except Exception as e:
        if logger:
            logger.error(f"Error cargando JSON {path}: {e}")
        return default


def save_json_safe(
    data: Any, path: Path, logger: Optional[PipelineLogger] = None
) -> bool:
    """Guarda JSON de forma segura"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if logger:
            logger.debug(f"JSON guardado: {path}")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Error guardando JSON {path}: {e}")
        return False


def validate_image_file(path: Path, supported_extensions: List[str]) -> bool:
    """Valida si un archivo es una imagen soportada"""
    if not path.exists() or not path.is_file():
        return False

    if path.suffix.lower() not in [ext.lower() for ext in supported_extensions]:
        return False

    # Verificar tamaño mínimo
    if path.stat().st_size < 1024:  # 1KB mínimo
        return False

    return True


def get_file_hash(path: Path) -> str:
    """Obtiene hash MD5 de un archivo"""
    import hashlib

    hash_md5 = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""


def cleanup_temp_files(
    temp_dir: Path, max_age_hours: int = 24, logger: Optional[PipelineLogger] = None
):
    """Limpia archivos temporales antiguos"""
    if not temp_dir.exists():
        return

    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0

    try:
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1

        if logger and deleted_count > 0:
            logger.info(f"Limpiados {deleted_count} archivos temporales")

    except Exception as e:
        if logger:
            logger.error(f"Error limpiando archivos temporales: {e}")


@contextmanager
def temp_directory(base_dir: Path, prefix: str = "avatar_temp"):
    """Context manager para directorio temporal"""
    import tempfile

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=base_dir))
        yield temp_dir
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


class SystemMonitor:
    """Monitor de recursos del sistema"""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Obtiene información de la GPU"""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            gpu_props = torch.cuda.get_device_properties(0)

            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": gpu_props.total_memory / (1024**3),
                "major": gpu_props.major,
                "minor": gpu_props.minor,
                "multi_processor_count": gpu_props.multi_processor_count,
            }
        except ImportError:
            return {"available": False, "error": "PyTorch not installed"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Obtiene información de memoria RAM"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent,
            }
        except ImportError:
            return {"error": "psutil not installed"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_disk_info(path: Path) -> Dict[str, Any]:
        """Obtiene información de espacio en disco"""
        try:
            stat = shutil.disk_usage(path)
            return {
                "total_gb": stat.total / (1024**3),
                "free_gb": stat.free / (1024**3),
                "used_gb": (stat.total - stat.free) / (1024**3),
                "used_percent": ((stat.total - stat.free) / stat.total) * 100,
            }
        except Exception as e:
            return {"error": str(e)}


def check_dependencies() -> Dict[str, bool]:
    """Verifica dependencias críticas del sistema"""
    dependencies = {
        "torch": False,
        "torchvision": False,
        "PIL": False,
        "cv2": False,
        "numpy": False,
        "pandas": False,
        "mtcnn": False,
        "rawpy": False,
        "imageio": False,
        "diffusers": False,
        "transformers": False,
        "accelerate": False,
        "safetensors": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    return dependencies


def estimate_processing_time(
    num_images: int,
    gpu_profile: Optional[str] = None,
    operation: str = "face_processing",
) -> Dict[str, float]:
    """Estima tiempo de procesamiento"""

    # Tiempos base por operación (segundos por imagen)
    base_times = {
        "face_processing": 2.0,
        "raw_conversion": 5.0,
        "dataset_preparation": 0.5,
        "lora_training_per_step": 0.001,  # Por step, no por imagen
    }

    # Multiplicadores por GPU
    gpu_multipliers = {
        "rtx_3050": 1.0,
        "rtx_3060": 0.9,
        "rtx_4060": 0.8,
        "rtx_4060_ti": 0.7,
        "low_end": 2.0,
        "high_end": 0.6,
    }

    base_time = base_times.get(operation, 1.0)
    multiplier = gpu_multipliers.get(gpu_profile, 1.0)

    total_seconds = num_images * base_time * multiplier

    return {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "formatted": format_time(total_seconds),
    }


def validate_client_structure(client_path: Path) -> Dict[str, bool]:
    """Valida la estructura de directorios de un cliente"""
    required_dirs = [
        "raw_mj",
        "raw_real",
        "processed",
        "dataset_lora",
        "rejected",
        "metadata",
        "training",
        "models",
        "samples",
        "output",
    ]

    structure = {}
    for dir_name in required_dirs:
        dir_path = client_path / dir_name
        structure[dir_name] = dir_path.exists() and dir_path.is_dir()

    return structure


def get_client_statistics(client_path: Path) -> Dict[str, Any]:
    """Obtiene estadísticas completas de un cliente"""
    if not client_path.exists():
        return {"error": "Client path does not exist"}

    stats = {
        "client_id": client_path.name,
        "created_date": datetime.fromtimestamp(client_path.stat().st_ctime).isoformat(),
        "last_modified": datetime.fromtimestamp(
            client_path.stat().st_mtime
        ).isoformat(),
        "structure": validate_client_structure(client_path),
        "file_counts": {},
        "total_size_mb": 0,
        "training_history": [],
        "latest_model": None,
    }

    # Contar archivos por directorio
    for subdir in [
        "raw_mj",
        "raw_real",
        "processed",
        "dataset_lora",
        "rejected",
        "models",
    ]:
        subdir_path = client_path / subdir
        if subdir_path.exists():
            if subdir == "models":
                files = list(subdir_path.glob("*.safetensors"))
            else:
                files = list(subdir_path.glob("*.*"))

            stats["file_counts"][subdir] = len(files)

            # Calcular tamaño total
            for file in files:
                if file.is_file():
                    stats["total_size_mb"] += file.stat().st_size / (1024 * 1024)

    # Modelo más reciente
    models_dir = client_path / "models"
    if models_dir.exists():
        models = list(models_dir.glob("*.safetensors"))
        if models:
            latest_model = max(models, key=lambda x: x.stat().st_mtime)
            stats["latest_model"] = {
                "name": latest_model.name,
                "size_mb": latest_model.stat().st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(
                    latest_model.stat().st_mtime
                ).isoformat(),
            }

    # Cargar historial de entrenamiento si existe
    config_file = client_path / "metadata" / "client_config.json"
    if config_file.exists():
        config_data = load_json_safe(config_file, {})
        stats["training_history"] = config_data.get("training_history", [])

    return stats
