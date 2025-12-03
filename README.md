# Programa para Reescalar Imágenes con IA

## Descripción

Este programa permite reescalar imágenes utilizando **Real-ESRGAN**, una red neuronal de última generación para super-resolución de imágenes. Aprovecha las capacidades de las tarjetas gráficas NVIDIA (CUDA) o AMD (Vulkan) para acelerar el proceso.

## Versiones Disponibles

### Nueva Versión Python (Recomendada)

La nueva versión usa Python con modelos actualizados de Real-ESRGAN para mejor calidad.

#### Requisitos

- Python 3.8 o superior
- Tarjeta gráfica NVIDIA con CUDA (recomendado) o CPU

#### Instalación

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el programa
python escalar.py
```

#### Uso con Interfaz Gráfica

```bash
python escalar.py
```

Esto abrirá una ventana donde puedes:
- Seleccionar una imagen o carpeta
- Elegir el modelo de escalado
- Configurar la escala (2x, 4x, 8x, 16x)
- Seleccionar el formato de salida

#### Uso por Línea de Comandos

```bash
# Escalar una imagen (x4 por defecto)
python escalar.py imagen.jpg

# Usar modelo para anime
python escalar.py imagen.jpg -m anime

# Escalar a 8x
python escalar.py imagen.jpg -s 8

# Escalar toda una carpeta
python escalar.py carpeta_imagenes/ -o carpeta_salida/

# Ver todas las opciones
python escalar.py --help
```

#### Modelos Disponibles

| Modelo | Escala | Descripción |
|--------|--------|-------------|
| `realesrgan-x4plus` | 4x | Fotos reales - Alta calidad |
| `realesrgan-x4plus-anime` | 4x | Optimizado para anime e ilustraciones |
| `realesrgan-x2plus` | 2x | Fotos reales - Más rápido |
| `realesr-general-x4v3` | 4x | Última versión - Mejor calidad |

---

### Versión Windows (Ejecutable)

Para usuarios que prefieren no instalar Python.

#### Requisitos

- Windows 10/11
- Tarjeta gráfica NVIDIA compatible con CUDA o AMD con Vulkan

#### Uso

1. **Descarga y Descomprime el Programa**
   - Descarga el archivo comprimido desde el repositorio y descomprímelo

2. **Ejecuta el Programa Principal**
   - Abre `Programa principal escalar imagen.exe`

   ![imagen](https://github.com/user-attachments/assets/d6874320-0d66-45a5-ab73-1287318bd09b)

3. **Selecciona la Imagen**
   - El programa abrirá una interfaz donde podrás seleccionar las imágenes

4. **Configura el Tamaño de Salida**
   - Establece el tamaño deseado para las imágenes reescaladas

5. **Inicia el Proceso**
   - Haz clic en "Reescalar" para comenzar

---

## Estructura del Proyecto

```
rescaladoImagenes/
├── escalar.py              # Punto de entrada principal (Python)
├── requirements.txt        # Dependencias Python
├── src/
│   ├── upscaler.py        # Módulo principal de escalado
│   ├── cli.py             # Interfaz de línea de comandos
│   └── gui.py             # Interfaz gráfica
├── models/                 # Modelos para versión Windows
├── models_python/          # Modelos descargados (Python)
└── Programa principal escalar imagen.exe  # Versión Windows
```

## Comparación de Versiones

| Característica | Versión Python | Versión Windows |
|----------------|----------------|-----------------|
| Modelos | Actualizados (2024) | Originales |
| Instalación | Requiere Python | Ejecutable directo |
| Calidad | Mayor | Buena |
| Personalización | Alta | Limitada |
| Soporte GPU | NVIDIA (CUDA) | NVIDIA/AMD (Vulkan) |
