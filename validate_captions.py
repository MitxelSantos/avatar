from pathlib import Path
import chardet

dataset_dir = Path("E:/Proyectos/Avatar/clients/Esoterico/training_data/10_Esoterico")

print("🔍 INSPECCIÓN DETALLADA DE CAPTIONS")
print("=" * 70)

# Ver primeros 10 captions
for i, txt_file in enumerate(sorted(dataset_dir.glob("*.txt"))[:10], 1):
    print(f"\n{'='*70}")
    print(f"📄 CAPTION #{i}: {txt_file.name}")
    print(f"{'='*70}")
    
    # Detectar encoding
    with open(txt_file, 'rb') as f:
        raw_bytes = f.read()
        detected = chardet.detect(raw_bytes)
        print(f"Encoding detectado: {detected['encoding']} (confianza: {detected['confidence']:.2%})")
    
    # Leer con UTF-8
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Longitud: {len(content)} caracteres")
        print(f"Contenido completo:")
        print(f"  '{content}'")
        
        # Mostrar bytes hex de primeros 50 chars
        print(f"\nBytes (hex) de primeros 50 chars:")
        print(f"  {content[:50].encode('utf-8').hex()}")
        
        # Contar palabras
        words = content.split(',')
        print(f"\nPalabras separadas por coma: {len(words)}")
        for j, word in enumerate(words[:5], 1):
            print(f"  {j}. '{word.strip()}'")
        
    except Exception as e:
        print(f"❌ ERROR LEYENDO: {e}")