# Upscaler de Alta Fidelidad - Version Python

Version avanzada del upscaler con modelos de ultima generacion para maxima fidelidad.

## Modelos Disponibles

| Modelo | Fidelidad | Velocidad | Uso Recomendado |
|--------|-----------|-----------|-----------------|
| `realesrgan` | ⭐⭐⭐⭐ | Rapido | Fotos generales |
| `realesrgan_anime` | ⭐⭐⭐⭐ | Rapido | Anime/ilustraciones |
| `swinir` | ⭐⭐⭐⭐⭐ | Medio | Alta fidelidad general |
| `hat` | ⭐⭐⭐⭐⭐+ | Lento | Maxima calidad (SOTA) |

## Requisitos

- Python 3.9+
- GPU NVIDIA con CUDA (recomendado) o CPU
- 8GB+ RAM (16GB recomendado para HAT)
- 4GB+ VRAM para GPU

## Instalacion

```bash
# 1. Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 2. Instalar PyTorch (con CUDA para GPU)
# Visita https://pytorch.org para el comando exacto segun tu sistema
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Imagen individual
```bash
# Real-ESRGAN (rapido, buena calidad)
python upscaler.py -i imagen.jpg -m realesrgan

# SwinIR (alta fidelidad)
python upscaler.py -i imagen.jpg -m swinir -s 4

# HAT (maxima calidad - estado del arte)
python upscaler.py -i imagen.jpg -m hat -s 4 -f png
```

### Carpeta completa
```bash
python upscaler.py -i carpeta_imagenes/ -m hat -s 4 -o carpeta_salida/
```

### Opciones
```
-i, --input    : Imagen o carpeta de entrada (requerido)
-o, --output   : Imagen o carpeta de salida (opcional)
-m, --model    : Modelo a usar (realesrgan, realesrgan_anime, swinir, hat)
-s, --scale    : Factor de escala (2 o 4, default: 4)
-f, --format   : Formato de salida (png o jpg, default: png)
--cpu          : Forzar uso de CPU
```

## Comparacion de Modelos

### Real-ESRGAN
- Buena calidad general
- Muy rapido con GPU
- Ideal para procesamiento por lotes

### SwinIR
- Basado en Swin Transformer
- Excelente preservacion de detalles
- Balance entre calidad y velocidad

### HAT (Hybrid Attention Transformer)
- Estado del arte (SOTA) en super-resolucion
- Maxima fidelidad posible
- Combina Channel Attention + Self-Attention + Overlapping Cross-Attention
- Requiere mas VRAM y tiempo

## Descargar Pesos de Modelos

Los pesos de Real-ESRGAN se descargan automaticamente.

Para HAT, descarga los pesos de:
https://github.com/XPixelGroup/HAT/releases

Y colocalos en la carpeta `weights/`:
- `HAT_x2.pth` para escala x2
- `HAT_x4.pth` para escala x4

## Estructura del Proyecto

```
python_version/
├── upscaler.py       # Script principal
├── requirements.txt  # Dependencias
├── models/
│   ├── __init__.py
│   ├── swinir.py     # Arquitectura SwinIR
│   └── hat.py        # Arquitectura HAT
└── weights/          # Pesos de modelos (se descargan automaticamente)
```

## Tips para Mejor Calidad

1. **Usa formato PNG** para evitar compresion con perdida
2. **HAT para fotos importantes** - tarda mas pero da mejores resultados
3. **Escala x4** generalmente da mejores resultados que x2 dos veces
4. **Mas VRAM = mejor** - permite procesar tiles mas grandes

## Troubleshooting

### Error de CUDA/GPU
```bash
# Usar CPU en su lugar
python upscaler.py -i imagen.jpg -m realesrgan --cpu
```

### Error de memoria (OOM)
- Usa una imagen mas pequena como entrada
- Usa `--cpu` para evitar limites de VRAM
- El script automaticamente procesa en tiles para ahorrar memoria

## Referencias

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [HAT](https://github.com/XPixelGroup/HAT)
