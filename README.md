# Avatar Pipeline - Sistema Profesional de Entrenamiento LoRA

Un sistema completo para procesar imágenes MidJourney y fotos reales, crear datasets balanceados y entrenar modelos LoRA SDXL de alta calidad para generación de avatares.

---

## 📋 PROMPT META - CONTEXTO COMPLETO DEL PROYECTO

**IMPORTANTE**: Este prompt meta está diseñado para que cualquier modelo de IA (incluyendo futuras versiones de Claude) pueda comprender instantáneamente el proyecto completo sin necesidad de explorar archivos.

### Propósito del Sistema
Sistema end-to-end para entrenar modelos LoRA personalizados para SDXL (Stable Diffusion XL) usando imágenes generadas por MidJourney y fotografías reales. El objetivo es crear avatares digitales de alta calidad que mantengan coherencia facial y estilística.

### Arquitectura del Sistema

**Módulos Principales:**

1. **avatar_pipeline.py** (Coordinador principal)
   - Orquesta todos los módulos
   - Gestiona menús interactivos
   - Maneja múltiples clientes
   - Estructura: `clients/{client_id}/{raw_mj,raw_real,processed,training_data,models,etc}`

2. **data_preprocessor.py** (Importación y metadata)
   - Procesa imágenes MidJourney con captura completa de metadata
   - Procesa fotos reales (incluye soporte RAW con rawpy/imageio)
   - Prepara datasets con estructura Kohya_ss: `training_data/{repeats}_{client_id}/`
   - Genera captions automáticos (.txt por cada imagen)
   - Distribución inteligente: 85% MJ / 15% Real por defecto

3. **image_processor.py** (Procesamiento facial)
   - Detección facial MTCNN (TensorFlow/Keras)
   - Recorte cuadrado 1024x1024 con padding inteligente
   - Control de calidad automatizado: confianza facial, brillo, contraste, nitidez
   - Parámetros QC optimizados para archivos RAW vs estándar

4. **lora_trainer.py** (Entrenamiento LoRA)
   - **CRÍTICO**: Usa `sdxl_train_network.py` (NO `train_network.py`)
   - Framework: Kohya_ss (sd-scripts)
   - Soporte SDXL exclusivo
   - Detección automática de GPU y optimizaciones
   - Perfiles para GTX 1650 4GB hasta RTX 4080
   - Comando clave: `--no_half_vae` (evita NaN en latents)

5. **config.py** (Configuración centralizada)
   - GPUProfile: optimizaciones por modelo de GPU
   - DatasetDistribution: ratios MJ/Real
   - QualityControlConfig: thresholds de QC
   - Training presets: quick/balanced/quality

6. **utils.py** (Utilidades)
   - PipelineLogger: logging estructurado
   - ProgressTracker: barra de progreso
   - Safe file operations
   - JSON load/save helpers

### Flujo de Trabajo Típico

```
1. Setup inicial → Crear cliente
2. Importar imágenes MJ → Captura metadata completa
3. Importar fotos reales → Análisis automático + conversión RAW
4. Procesamiento facial → MTCNN + QC → 1024x1024 PNG
5. Preparar dataset LoRA → Estructura Kohya_ss + captions
6. Configurar entrenamiento → Seleccionar preset según GPU
7. Entrenar modelo → sdxl_train_network.py + monitoring
8. Modelo final → .safetensors listo para ComfyUI/A1111
```

### Problemas Resueltos y Lecciones Aprendidas

**Problema 1: "0 train images" en Kohya_ss**
- Causa: Estructura de directorios incorrecta
- Solución: Usar `training_data/{repeats}_{client_id}/` como formato
- Kohya_ss espera directorio PADRE en `--train_data_dir`

**Problema 2: Script incorrecto para SDXL**
- Causa: Usar `train_network.py` en lugar de `sdxl_train_network.py`
- Solución: Kohya_ss tiene scripts separados por modelo base
- SD 1.5 → `train_network.py`
- SDXL → `sdxl_train_network.py`

**Problema 3: NaN detected in latents**
- Causa: VAE en fp16 genera valores inválidos en ciertas imágenes
- Solución: Agregar flag `--no_half_vae` al comando de entrenamiento
- Trade-off: Usa más VRAM pero previene errores

**Problema 4: Warnings repetitivos (xFormers, Triton, TensorFlow)**
- Causa: Versiones de PyTorch/CUDA/xFormers no coinciden perfectamente
- Solución: Suprimir con variables de entorno:
  ```python
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  os.environ["PYTHONWARNINGS"] = "ignore"
  ```
- Impacto: Los warnings NO afectan funcionalidad, solo ruido visual

**Problema 5: Backslashes en Windows**
- Causa: `\` no funciona para continuación de línea en CMD
- Solución: Usar `^` en .bat o escribir comando en una sola línea

### Dependencias Críticas

**PyTorch:**
- Versión: 2.1.2+cu118 (CUDA 11.8)
- DEBE instalarse PRIMERO antes de requirements.txt
- Comando: `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`

**Kohya_ss:**
- Repositorio: https://github.com/kohya-ss/sd-scripts
- Versión actual usa `sdxl_train_network.py`
- NO incluir en requirements.txt, se clona separadamente

**MTCNN:**
- Detección facial robusta
- Requiere TensorFlow (CPU-only para liberar VRAM)
- `tensorflow-cpu==2.15.1` y `mtcnn==0.1.1`

**rawpy/imageio:**
- Soporte para archivos RAW (NEF, CR2, ARW, DNG, etc.)
- Conversión automática a JPEG temporal para procesamiento

### Metadata MidJourney Capturada

Información preservada por grupo UUID (4 variaciones):
- Prompt maestro (base experimental)
- Prompts específicos por grupo
- Parámetros: `--v`, `--q`, `--s`, `--chaos`, `--ar`, `--style`, `--ow` (150-170), `--variety`, `--weird`
- Características auto-detectadas: iluminación, expresión, físicas

### Estructura de Directorios Cliente

```
clients/{client_id}/
├── raw_mj/              # Imágenes MJ originales
├── raw_real/            # Fotos reales originales
├── processed/           # Recortes 1024x1024 aprobados por QC
├── rejected/            # Imágenes rechazadas por QC
├── training_data/       # PADRE para Kohya_ss
│   └── {repeats}_{client_id}/  # Subdirectorio con formato Kohya
│       ├── *.png        # Imágenes procesadas
│       ├── *.txt        # Captions (uno por imagen)
│       └── dataset_info.json
├── metadata/            # CSVs, JSONs de tracking
├── training/            # Configuraciones y logs
│   ├── lora_config.json
│   └── logs/           # TensorBoard logs
├── models/             # Modelos .safetensors entrenados
├── samples/            # Muestras generadas
└── output/             # Exports finales
```

### Parámetros de Entrenamiento Típicos

**GPU 8GB (RTX 3060/4060):**
```
--network_dim 96
--network_alpha 48
--resolution 1024,1024
--train_batch_size 1
--gradient_accumulation_steps 2
--mixed_precision fp16
--optimizer_type AdamW8bit
```

**GPU 4GB (GTX 1650):**
```
--network_dim 32
--network_alpha 16
--resolution 768,768
--train_batch_size 1
--gradient_accumulation_steps 8
--mixed_precision fp16
--optimizer_type AdamW8bit
--lowvram
```

### Comando de Entrenamiento Completo (Ejemplo)

```bash
cd kohya_ss

python sdxl_train_network.py \
  --train_data_dir "../clients/ClienteX/training_data" \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --caption_extension ".txt" \
  --network_module "networks.lora" \
  --network_dim 96 \
  --network_alpha 48 \
  --max_train_steps 2000 \
  --learning_rate 0.0001 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --optimizer_type AdamW8bit \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --cache_latents \
  --no_half_vae \
  --output_dir "../clients/ClienteX/models" \
  --output_name "ClienteX_avatar_v1" \
  --save_model_as safetensors \
  --save_every_n_steps 500 \
  --logging_dir "../clients/ClienteX/training/logs"
```

### Trigger Word y Uso del Modelo

**Trigger:** Nombre del cliente (ej: `ClienteX`)

**Prompts recomendados:**
```
portrait of ClienteX, professional headshot, studio lighting, detailed face
ClienteX smiling, natural lighting, high quality photography
close-up of ClienteX, artistic portrait, dramatic lighting
```

**Configuración:**
- Weight: 0.7 - 1.0
- Steps: 20-30
- CFG Scale: 7-8
- Sampler: Euler A / DPM++ 2M

### Troubleshooting Común

**Error: No face detected**
- Verificar calidad de imagen
- Reducir `face_confidence_threshold` en config
- Asegurar rostro claramente visible

**Error: CUDA out of memory**
- Reducir `resolution` (1024→768)
- Reducir `network_dim` (96→64→32)
- Aumentar `gradient_accumulation_steps`
- Activar optimizaciones de memoria

**Error: Training hangs at epoch increment**
- Múltiples procesos hijo lanzándose
- Verificar que solo haya un proceso Python activo
- Reiniciar y volver a ejecutar

### Información de Versiones

**Versión del Sistema:** 4.0
**Última actualización:** Septiembre 2025
**Python:** 3.11.8
**PyTorch:** 2.1.2+cu118
**CUDA:** 11.8
**Kohya_ss:** Última versión (sd-scripts main branch)

### Próximas Mejoras Planificadas

1. Generación automática de muestras durante entrenamiento
2. Integración con APIs de generación (A1111, ComfyUI)
3. Fine-tuning automático de hiperparámetros
4. Soporte para múltiples rostros en una imagen
5. Dashboard web para monitoreo en tiempo real

---

## 🎯 Características Principales

- **Procesamiento facial avanzado** con detección MTCNN y control de calidad automatizado
- **Captura completa de metadata MidJourney** con preservación de prompts y parámetros
- **Balance inteligente de datos** (85% MJ / 15% Real) optimizado para avatares sintéticos
- **Entrenamiento LoRA para SDXL** optimizado desde 4GB hasta 24GB VRAM
- **Interfaz de menús interactiva** con flujo guiado paso a paso
- **Gestión profesional de proyectos** con múltiples clientes
- **Soporte completo para archivos RAW** (NEF, CR2, ARW, DNG, etc.)

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **GPU:** GTX 1650 4GB o superior (recomendado: RTX 3060 8GB+)
- **RAM:** 16GB (recomendado: 32GB)
- **Almacenamiento:** 50GB libres para datasets y modelos
- **Sistema:** Windows 10/11, Linux, macOS

### Software
- **Python:** 3.11.x (CRÍTICO: 3.11, no 3.12)
- **CUDA:** 11.8 (para PyTorch 2.1.2+cu118)
- **Git:** Para clonado de repositorios

## 🚀 Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd avatar-pipeline
```

### 2. Crear Entorno Virtual
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Instalar PyTorch PRIMERO (CRÍTICO)
```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (alternativa)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 5. Ejecutar Setup Inicial
```bash
python avatar_pipeline.py
```
Selecciona opción 1: "Setup inicial del proyecto"

## 📖 Guía de Uso

### Flujo de Trabajo Recomendado

#### 1. **Setup Inicial**
```bash
python avatar_pipeline.py
```
- Ejecuta setup del proyecto
- Configura estructura de directorios

#### 2. **Crear Cliente**
- Menú Principal → Gestión de clientes → Crear nuevo cliente
- Ingresa nombre único del cliente (ej: "cliente01_nombre")

#### 3. **Importar Imágenes MidJourney**
- Selecciona cliente → Cargar imágenes MidJourney
- **Captura de Metadata Completa:**
  - Prompt maestro (base experimental exitoso)
  - Prompts específicos por grupo de 4 imágenes
  - Parámetros automáticos detectados del filename

#### 4. **Importar Fotos Reales** (Opcional)
- Selecciona cliente → Cargar fotos reales
- Compatible con archivos RAW y formatos estándar
- Análisis automático de características

#### 5. **Procesamiento Facial**
- Detección MTCNN con alta precisión
- Recorte cuadrado 1024x1024
- Control de calidad automatizado

#### 6. **Preparar Dataset LoRA**
- Estructura Kohya_ss automática: `{repeats}_{client_id}/`
- Balance 85% MJ / 15% Real (configurable)
- Captions ricos con metadata preservada

#### 7. **Entrenamiento LoRA**
- Configuración automática según GPU detectada
- Perfiles optimizados desde GTX 1650 4GB
- Monitoring en tiempo real con TensorBoard

## ⚙️ Configuración Avanzada

### Parámetros de Calidad Facial
```json
{
  "face_confidence_threshold": 0.85,
  "face_padding_factor": 1.6,
  "min_brightness": 40,
  "max_brightness": 220,
  "min_contrast": 25,
  "blur_threshold": 100
}
```

### Perfiles de Entrenamiento LoRA

| GPU | VRAM | Dim | Steps/h | Resolución | Tiempo 2000 steps |
|-----|------|-----|---------|------------|-------------------|
| GTX 1650 | 4GB | 32 | 175 | 768 | ~11h |
| RTX 3050 | 8GB | 96 | 400 | 1024 | ~5h |
| RTX 3060 | 8GB | 96 | 450 | 1024 | ~4.5h |
| RTX 4060 | 8GB | 128 | 500 | 1024 | ~4h |
| RTX 4060 Ti | 16GB | 128 | 600 | 1024 | ~3.5h |

### Optimizaciones para GPUs con Poca VRAM

**GTX 1650 4GB:**
```python
{
  "network_dim": 32,
  "network_alpha": 16,
  "resolution": 768,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "mixed_precision": "fp16",
  "optimizer": "AdamW8bit",
  "lowvram": True,
  "medvram": True
}
```

## 📊 Metadata MidJourney Capturada

### Información Preservada
- **Prompt maestro** del avatar base experimental
- **Prompts específicos** por grupo UUID (4 variaciones)
- **Parámetros técnicos completos:**
  - `--v` (version)
  - `--q` (quality) 
  - `--s` (stylize)
  - `--chaos`
  - `--ar` (aspect ratio)
  - `--style` (raw/standard)
  - `--ow` (omni weight: 150-170)
  - `--variety`
  - `--weird` (weirdness)

### Características Auto-detectadas
- **Iluminación:** studio, natural, dramatic, soft, cinematic
- **Expresión:** confident, serene, trustworthy, wise, professional
- **Características físicas:** clean_shaven, stubble, well_groomed, refined

## 🎨 Uso del Modelo LoRA Entrenado

### Trigger Word
```
Nombre del cliente (ej: "cliente01_nombre")
```

### Prompts Recomendados
```
portrait of [cliente_id], professional headshot, studio lighting, detailed face

[cliente_id] smiling, natural lighting, high quality photography  

close-up of [cliente_id], artistic portrait, dramatic lighting
```

### Configuración Recomendada
- **Weight:** 0.7 - 1.0
- **Steps:** 20-30
- **CFG Scale:** 7-8
- **Sampler:** Euler A / DPM++ 2M
- **Modelo base:** SDXL 1.0

### Compatibilidad
- ✅ ComfyUI
- ✅ Automatic1111
- ✅ Diffusers (Python)
- ✅ Invoke AI

## 🔧 Solución de Problemas

### Errores Comunes

**"MTCNN not available"**
```bash
pip install mtcnn tensorflow-cpu
```

**"CUDA out of memory"**
- Cierra otras aplicaciones
- Reduce resolution a 768
- Reduce network_dim a 64 o 32
- Activa todas las optimizaciones de memoria

**"No face detected"**
- Verifica calidad de las imágenes
- Reduce face_confidence_threshold a 0.75
- Asegúrate de que el rostro sea claramente visible

**"NaN detected in latents"**
- Agrega flag `--no_half_vae` al comando de entrenamiento
- Este problema es específico de SDXL en fp16

**"0 train images" en Kohya_ss**
- Verifica estructura de directorios: debe ser `training_data/{repeats}_{client}/`
- Asegúrate de usar `sdxl_train_network.py` (no `train_network.py`)
- Verifica que existan archivos .png y .txt en el subdirectorio

**"Script incorrecto para SDXL"**
- NO uses `train_network.py` (es para SD 1.5)
- USA `sdxl_train_network.py` para SDXL
- Verifica en `kohya_ss/` que existe `sdxl_train_network.py`

### Optimización de Rendimiento

**Para GPUs con poca VRAM (<8GB):**
- Resolution: 768 en lugar de 1024
- Network dim: 32-64 en lugar de 96-128
- Gradient accumulation: 4-8 en lugar de 1-2
- Optimizer: AdamW8bit obligatorio
- Flags: `--lowvram --medvram`

**Para entrenamiento más rápido (8GB+):**
- Resolution: 1024
- Network dim: 96-128
- Batch size: 1-2
- Learning rate: 0.00012 (en lugar de 0.0001)
- Gradient accumulation: 1-2

## 📈 Monitoreo del Entrenamiento

### TensorBoard
```bash
tensorboard --logdir clients/[cliente_id]/training/logs
```

### Archivos de Progreso
- **Checkpoints:** `clients/[cliente_id]/models/*.safetensors`
- **Samples:** `clients/[cliente_id]/training/samples/*.png`
- **Logs:** `clients/[cliente_id]/training/logs/`

## 🤝 Contribución

### Estructura para Nuevas Características
1. Crear módulo especializado
2. Integrar con `avatar_pipeline.py`
3. Actualizar tests y documentación
4. Seguir convenciones de nomenclatura

### Convenciones de Código
- **Nombres de archivo:** snake_case
- **Clases:** PascalCase
- **Funciones:** snake_case
- **Constantes:** UPPER_CASE

## 📜 Licencia

MIT License - Ver archivo LICENSE

## 🙏 Agradecimientos

- **Kohya_ss** por el framework de entrenamiento LoRA
- **MTCNN** por la detección facial robusta
- **Stability AI** por SDXL base model
- **Comunidad de Diffusers** por las optimizaciones

## 📞 Soporte

Para reportar bugs o solicitar características:
1. Revisa la sección de solución de problemas
2. Revisa el PROMPT META arriba para contexto completo
3. Crea un issue con información detallada
4. Incluye logs relevantes y configuración del sistema

---

**Avatar Pipeline v1.0** - Sistema Profesional de Entrenamiento LoRA para Avatares SDXL

*Última actualización: Septiembre 2025*