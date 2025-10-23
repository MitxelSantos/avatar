#!/usr/bin/env python3
"""
fix_captions_variability.py - Agrega variabilidad a captions idénticos
"""

from pathlib import Path
import random

def add_caption_variability(dataset_dir: Path):
    """Agrega variaciones aleatorias a captions para mejorar entrenamiento"""
    
    dataset_dir = Path(dataset_dir)
    
    # Variaciones posibles para agregar diversidad
    variations = {
        "lighting": [
            "studio lighting",
            "natural light",
            "professional lighting setup",
            "dramatic lighting",
            "soft diffused light",
            "three point lighting",
        ],
        "angle": [
            "front view",
            "slight angle",
            "three quarter view",
            "direct gaze",
            "neutral perspective",
        ],
        "expression": [
            "confident expression",
            "serene demeanor",
            "calm presence",
            "trustworthy appearance",
            "professional composure",
        ],
        "background": [
            "neutral background",
            "clean background",
            "professional backdrop",
            "simple background",
        ],
        "quality": [
            "high quality",
            "detailed face",
            "sharp focus",
            "professional photography",
            "crisp details",
        ]
    }
    
    caption_files = sorted(dataset_dir.glob("*.txt"))
    
    print(f"\n🔧 AGREGANDO VARIABILIDAD A CAPTIONS")
    print("=" * 60)
    print(f"Total de captions: {len(caption_files)}")
    
    for i, caption_path in enumerate(caption_files, 1):
        # Caption base (mantener trigger word)
        base_caption = "Esoterico, portrait of Mexican mestizo brujo moderno"
        
        # Agregar variaciones aleatorias
        selected_variations = []
        
        # Seleccionar 1-2 variaciones de cada categoría
        for category, options in variations.items():
            num_to_select = random.randint(1, min(2, len(options)))
            selected = random.sample(options, num_to_select)
            selected_variations.extend(selected)
        
        # Shuffle para más aleatoriedad
        random.shuffle(selected_variations)
        
        # Construir caption final
        final_caption = f"{base_caption}, {', '.join(selected_variations)}"
        
        # Guardar
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(final_caption)
        
        if i <= 5 or i % 20 == 0:
            print(f"\n📝 Caption #{i}: {caption_path.name}")
            print(f"   {final_caption[:100]}...")
    
    print(f"\n✅ {len(caption_files)} captions actualizados con variabilidad")


if __name__ == "__main__":
    # Configurar seed para reproducibilidad pero con variedad
    random.seed(42)
    
    dataset_path = Path("E:/Proyectos/Avatar/clients/Esoterico/training_data/10_Esoterico")
    add_caption_variability(dataset_path)