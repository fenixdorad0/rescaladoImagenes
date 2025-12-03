"""
Módulo principal para escalado de imágenes con Real-ESRGAN
Usa modelos actualizados para mejor calidad de imagen
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Literal
from PIL import Image

# Real-ESRGAN imports
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# Modelos disponibles y sus configuraciones
MODELOS_DISPONIBLES = {
    "realesrgan-x4plus": {
        "descripcion": "Modelo general para fotos reales - Alta calidad",
        "escala": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
    },
    "realesrgan-x4plus-anime": {
        "descripcion": "Optimizado para anime e ilustraciones",
        "escala": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "num_block": 6,
        "num_feat": 64,
        "num_grow_ch": 32,
    },
    "realesrgan-x2plus": {
        "descripcion": "Modelo general 2x - Más rápido",
        "escala": 2,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
    },
    "realesr-general-x4v3": {
        "descripcion": "Modelo general v3 - Última versión, mejor calidad",
        "escala": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
    },
}


class EscaladorImagenes:
    """Clase principal para escalar imágenes usando Real-ESRGAN"""

    def __init__(
        self,
        modelo: str = "realesrgan-x4plus",
        gpu_id: int = 0,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half_precision: bool = True,
    ):
        """
        Inicializa el escalador de imágenes.

        Args:
            modelo: Nombre del modelo a usar (ver MODELOS_DISPONIBLES)
            gpu_id: ID de la GPU a usar (0 para la primera, -1 para CPU)
            tile: Tamaño de tile para procesar (0 = sin tiles, usa más VRAM)
            tile_pad: Padding entre tiles
            pre_pad: Pre-padding de la imagen
            half_precision: Usar FP16 para menor uso de VRAM
        """
        self.modelo_nombre = modelo
        self.gpu_id = gpu_id
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half_precision = half_precision
        self.upsampler = None

        self._cargar_modelo()

    def _obtener_ruta_modelo(self) -> Path:
        """Obtiene la ruta donde se guardan los modelos"""
        ruta_modelos = Path(__file__).parent.parent / "models_python"
        ruta_modelos.mkdir(exist_ok=True)
        return ruta_modelos

    def _descargar_modelo(self, config: dict) -> str:
        """Descarga el modelo si no existe localmente"""
        import requests
        from tqdm import tqdm

        ruta_modelos = self._obtener_ruta_modelo()
        nombre_archivo = config["url"].split("/")[-1]
        ruta_archivo = ruta_modelos / nombre_archivo

        if ruta_archivo.exists():
            print(f"Modelo encontrado: {nombre_archivo}")
            return str(ruta_archivo)

        print(f"Descargando modelo: {nombre_archivo}...")
        response = requests.get(config["url"], stream=True)
        total = int(response.headers.get('content-length', 0))

        with open(ruta_archivo, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Modelo descargado: {ruta_archivo}")
        return str(ruta_archivo)

    def _cargar_modelo(self):
        """Carga el modelo de Real-ESRGAN"""
        if self.modelo_nombre not in MODELOS_DISPONIBLES:
            raise ValueError(
                f"Modelo '{self.modelo_nombre}' no disponible. "
                f"Opciones: {list(MODELOS_DISPONIBLES.keys())}"
            )

        config = MODELOS_DISPONIBLES[self.modelo_nombre]
        ruta_modelo = self._descargar_modelo(config)

        # Crear la red neuronal
        modelo = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config["num_feat"],
            num_block=config["num_block"],
            num_grow_ch=config["num_grow_ch"],
            scale=config["escala"]
        )

        # Inicializar el upsampler
        self.upsampler = RealESRGANer(
            scale=config["escala"],
            model_path=ruta_modelo,
            model=modelo,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half_precision,
            gpu_id=self.gpu_id if self.gpu_id >= 0 else None
        )

        self.escala = config["escala"]
        print(f"Modelo cargado: {self.modelo_nombre} ({config['descripcion']})")

    def escalar_imagen(
        self,
        ruta_entrada: str,
        ruta_salida: Optional[str] = None,
        escala_objetivo: Optional[int] = None,
        formato_salida: str = "png"
    ) -> str:
        """
        Escala una imagen.

        Args:
            ruta_entrada: Ruta de la imagen de entrada
            ruta_salida: Ruta de salida (opcional, se genera automáticamente)
            escala_objetivo: Escala objetivo (2, 4, 8, 16). Si es mayor que el modelo,
                           se aplica múltiples veces
            formato_salida: Formato de salida (png, jpg, webp)

        Returns:
            Ruta del archivo de salida
        """
        # Validar entrada
        ruta_entrada = Path(ruta_entrada)
        if not ruta_entrada.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_entrada}")

        # Leer imagen
        img = cv2.imread(str(ruta_entrada), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {ruta_entrada}")

        # Determinar escala objetivo
        if escala_objetivo is None:
            escala_objetivo = self.escala

        # Aplicar escalado (múltiples pasadas si es necesario)
        resultado = img
        escala_aplicada = 1

        while escala_aplicada < escala_objetivo:
            resultado, _ = self.upsampler.enhance(resultado, outscale=self.escala)
            escala_aplicada *= self.escala

        # Generar ruta de salida si no se especifica
        if ruta_salida is None:
            nombre_base = ruta_entrada.stem
            ruta_salida = ruta_entrada.parent / f"{nombre_base}_x{escala_aplicada}.{formato_salida}"
        else:
            ruta_salida = Path(ruta_salida)

        # Guardar imagen
        cv2.imwrite(str(ruta_salida), resultado)
        print(f"Imagen guardada: {ruta_salida}")

        return str(ruta_salida)

    def escalar_carpeta(
        self,
        carpeta_entrada: str,
        carpeta_salida: Optional[str] = None,
        escala_objetivo: Optional[int] = None,
        formato_salida: str = "png",
        extensiones: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ) -> list:
        """
        Escala todas las imágenes en una carpeta.

        Args:
            carpeta_entrada: Carpeta con imágenes de entrada
            carpeta_salida: Carpeta de salida (opcional)
            escala_objetivo: Escala objetivo
            formato_salida: Formato de salida
            extensiones: Extensiones de archivo a procesar

        Returns:
            Lista de rutas de archivos procesados
        """
        from tqdm import tqdm

        carpeta_entrada = Path(carpeta_entrada)
        if not carpeta_entrada.is_dir():
            raise NotADirectoryError(f"No es una carpeta: {carpeta_entrada}")

        # Crear carpeta de salida
        if carpeta_salida is None:
            carpeta_salida = carpeta_entrada / "escaladas"
        else:
            carpeta_salida = Path(carpeta_salida)
        carpeta_salida.mkdir(exist_ok=True)

        # Buscar imágenes
        imagenes = [
            f for f in carpeta_entrada.iterdir()
            if f.suffix.lower() in extensiones
        ]

        if not imagenes:
            print("No se encontraron imágenes para procesar")
            return []

        print(f"Procesando {len(imagenes)} imágenes...")
        resultados = []

        for img_path in tqdm(imagenes, desc="Escalando"):
            try:
                nombre_salida = f"{img_path.stem}_escalada.{formato_salida}"
                ruta_salida = carpeta_salida / nombre_salida

                self.escalar_imagen(
                    str(img_path),
                    str(ruta_salida),
                    escala_objetivo,
                    formato_salida
                )
                resultados.append(str(ruta_salida))
            except Exception as e:
                print(f"Error procesando {img_path.name}: {e}")

        print(f"Completado: {len(resultados)}/{len(imagenes)} imágenes procesadas")
        return resultados


def listar_modelos():
    """Muestra los modelos disponibles"""
    print("\n=== Modelos disponibles ===\n")
    for nombre, config in MODELOS_DISPONIBLES.items():
        print(f"  {nombre}")
        print(f"    Escala: {config['escala']}x")
        print(f"    {config['descripcion']}")
        print()


if __name__ == "__main__":
    listar_modelos()
