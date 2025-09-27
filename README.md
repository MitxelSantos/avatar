# Avatar Pipeline - Sistema Profesional de Entrenamiento LoRA

Un sistema completo para procesar imágenes MidJourney y fotos reales, crear datasets balanceados y entrenar modelos LoRA de alta calidad para generación de avatares.

## 🎯 Características Principales

- **Procesamiento facial avanzado** con detección MTCNN y control de calidad automatizado
- **Captura completa de metadata MidJourney** con preservación de prompts y parámetros
- **Balance inteligente de datos** (85% MJ / 15% Real) optimizado para deepfacelive
- **Entrenamiento LoRA para SDXL** optimizado para GPUs de 6GB VRAM
- **Interfaz de menús interactiva** con flujo guiado paso a paso
- **Gestión profesional de proyectos** con múltiples clientes

## 📋 Requisitos del Sistema

### Hardware Mínimo
- **GPU:** RTX 3050 6GB o superior (recomendado: RTX 3060 8GB+)
- **RAM:** 16GB (recomendado: 32GB)
- **Almacenamiento:** 50GB libres para datasets y modelos
- **Sistema:** Windows 10/11, Linux, macOS

### Software
- **Python:** 3.8 - 3.11
- **CUDA:** 11.8 o 12.1
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

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar Setup Inicial
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
- Ingresa nombre único del cliente (ej: "cliente01_brujo_catemaco")

#### 3. **Importar Imágenes MidJourney**
- Selecciona cliente → Cargar imágenes MidJourney
- **Captura de Metadata Completa:**
  - Prompt maestro (base experimental exitoso)
  - Prompts específicos por grupo de 4 imágenes
  - Parámetros automáticos: version, quality, stylize, chaos, aspect_ratio, style, omni_weight, variety, weirdness

#### 4. **Importar Fotos Reales** (Opcional)
- Selecciona cliente → Cargar fotos reales
- Para compatibilidad con deepfacelive
- Análisis automático de características

#### 5. **Procesamiento Facial**
- Detección MTCNN con alta precisión
- Recorte cuadrado 1024x1024
- Control de calidad automatizado:
  - Confianza facial: 0.85+
  - Brillo: 40-220
  - Contraste: 25+
  - Nitidez: 100+

#### 6. **Entrenamiento LoRA**
- **Perfil Máxima Calidad:** 3000 steps, 6-8 horas
- **Optimizado para SDXL** en 6GB VRAM
- **Balance profesional:** 85% MJ / 15% Real
- **Captions ricos** con metadata preservada

## 🏗️ Arquitectura del Sistema

### Módulos Principales

```
avatar_pipeline.py          # Coordinador principal y menús
├── image_processor.py      # Procesamiento facial MTCNN
├── data_preprocessor.py    # Metadata MJ + análisis fotos reales  
└── lora_trainer.py        # Entrenamiento LoRA con Kohya_ss
```

### Estructura de Proyecto

```
avatar-pipeline/
├── clients/                        # Datos de clientes
│   └── [cliente_id]/
│       ├── raw_mj/                 # Imágenes MJ originales
│       ├── raw_real/               # Fotos reales originales
│       ├── processed/              # Imágenes procesadas 1024x1024
│       ├── dataset_lora/           # Dataset final + captions
│       ├── rejected/               # Imágenes rechazadas por QC
│       ├── metadata/               # Logs, CSVs, configuraciones
│       ├── training/               # Logs y configuración entrenamiento
│       ├── models/                 # Modelos LoRA entrenados
│       └── output/                 # Exports y resultados finales
├── training/                       # Configuraciones globales
└── kohya_ss/                      # Framework de entrenamiento (auto-instalado)
```

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

| Perfil | Steps | Learning Rate | Tiempo | Calidad |
|--------|-------|---------------|---------|---------|
| Máxima Calidad | 3000 | 0.00008 | 6-8h | ⭐⭐⭐⭐⭐ |
| Equilibrado | 2000 | 0.0001 | 4-6h | ⭐⭐⭐⭐ |
| Rápido | 1500 | 0.00012 | 2-4h | ⭐⭐⭐ |

### Optimizaciones para 6GB VRAM
- Mixed precision (fp16)
- Gradient checkpointing
- Cache latents y text encoder
- AdamW8bit optimizer
- XFormers memory efficient attention
- Batch size: 1 con gradient accumulation: 4

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
Nombre del cliente (ej: "cliente01_brujo_catemaco")
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

### Compatibilidad
- ✅ ComfyUI
- ✅ Automatic1111
- ✅ Diffusers (Python)
- ✅ Invoke AI

## 🔧 Solución de Problemas

### Errores Comunes

**"MTCNN not available"**
```bash
pip install mtcnn tensorflow
```

**"CUDA out of memory"**
- Cierra otras aplicaciones
- Reduce batch_size a 1
- Activa todas las optimizaciones de memoria

**"No face detected"**
- Verifica calidad de las imágenes
- Reduce face_confidence_threshold
- Asegúrate de que el rostro sea claramente visible

**"Kohya_ss setup failed"**
```bash
# Instalación manual
git clone https://github.com/kohya-ss/sd-scripts.git kohya_ss
cd kohya_ss
pip install -r requirements.txt
```

### Optimización de Rendimiento

**Para GPUs con poca VRAM (<8GB):**
- Activa todas las optimizaciones de memoria
- Usa batch_size: 1
- Considera usar gradient_accumulation_steps: 4-8

**Para entrenamiento más rápido:**
- Reduce dataset_repeats
- Usa learning_rate más alto (0.00012)
- Reduce max_train_steps

## 📈 Monitoreo del Entrenamiento

### TensorBoard
```bash
tensorboard --logdir clients/[cliente_id]/training/logs
```

### Archivos de Progreso
- **Checkpoints:** `clients/[cliente_id]/models/*.safetensors`
- **Muestras:** `clients/[cliente_id]/training/samples/*.png`
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

[Especificar licencia aquí]

## 🙏 Agradecimientos

- **Kohya_ss** por el framework de entrenamiento LoRA
- **MTCNN** por la detección facial robusta
- **Stability AI** por SDXL base model
- **Comunidad de Diffusers** por las optimizaciones

## 📞 Soporte

Para reportar bugs o solicitar características:
1. Revisa la sección de solución de problemas
2. Crea un issue con información detallada
3. Incluye logs relevantes y configuración del sistema

---

**Avatar Pipeline v2.0** - Sistema Profesional de Entrenamiento LoRA para Avatares