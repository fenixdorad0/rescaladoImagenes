"""
Versión simplificada del escalador de imágenes
Usa el ejecutable realesrgan-ncnn-vulkan incluido en el proyecto
No requiere instalación de dependencias pesadas
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# Modelos disponibles
MODELOS_DISPONIBLES = {
    "realesrgan-x4plus": {
        "descripcion": "Modelo general para fotos reales - Alta calidad",
        "escala": 4,
    },
    "realesrgan-x4plus-anime": {
        "descripcion": "Optimizado para anime e ilustraciones",
        "escala": 4,
    },
    "realesr-animevideov3": {
        "descripcion": "Para video anime - Soporta x2, x3, x4",
        "escala": 4,
    },
}


class EscaladorImagenes:
    """Clase para escalar imágenes usando realesrgan-ncnn-vulkan"""

    def __init__(
        self,
        modelo: str = "realesrgan-x4plus",
        gpu_id: int = 0,
    ):
        """
        Inicializa el escalador de imágenes.

        Args:
            modelo: Nombre del modelo a usar
            gpu_id: ID de la GPU a usar (-1 para CPU)
        """
        self.modelo = modelo
        self.gpu_id = gpu_id
        self.ruta_base = Path(__file__).parent.parent

        # Buscar el ejecutable
        self.ejecutable = self._encontrar_ejecutable()
        if not self.ejecutable:
            raise FileNotFoundError(
                "No se encontró realesrgan-ncnn-vulkan. "
                "Asegúrate de que esté en la carpeta del proyecto."
            )

        self.escala = MODELOS_DISPONIBLES.get(modelo, {}).get("escala", 4)
        print(f"Escalador inicializado con modelo: {modelo}")

    def _encontrar_ejecutable(self) -> Optional[Path]:
        """Busca el ejecutable de realesrgan"""
        # Nombres posibles del ejecutable
        nombres = [
            "realesrgan-ncnn-vulkan.exe",
            "realesrgan-ncnn-vulkan",
        ]

        for nombre in nombres:
            ruta = self.ruta_base / nombre
            if ruta.exists():
                return ruta

        # Buscar en PATH
        ejecutable = shutil.which("realesrgan-ncnn-vulkan")
        if ejecutable:
            return Path(ejecutable)

        return None

    def escalar_imagen(
        self,
        ruta_entrada: str,
        ruta_salida: Optional[str] = None,
        escala: Optional[int] = None,
        formato_salida: str = "png"
    ) -> str:
        """
        Escala una imagen.

        Args:
            ruta_entrada: Ruta de la imagen de entrada
            ruta_salida: Ruta de salida (opcional)
            escala: Factor de escala (2, 3, 4)
            formato_salida: Formato de salida (png, jpg, webp)

        Returns:
            Ruta del archivo de salida
        """
        ruta_entrada = Path(ruta_entrada)
        if not ruta_entrada.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_entrada}")

        # Determinar escala
        if escala is None:
            escala = self.escala

        # Generar ruta de salida
        if ruta_salida is None:
            nombre_base = ruta_entrada.stem
            ruta_salida = ruta_entrada.parent / f"{nombre_base}_x{escala}.{formato_salida}"
        else:
            ruta_salida = Path(ruta_salida)

        # Construir comando
        cmd = [
            str(self.ejecutable),
            "-i", str(ruta_entrada),
            "-o", str(ruta_salida),
            "-s", str(escala),
            "-n", self.modelo,
            "-f", formato_salida,
        ]

        # Agregar GPU si no es -1
        if self.gpu_id >= 0:
            cmd.extend(["-g", str(self.gpu_id)])

        print(f"Procesando: {ruta_entrada.name}...")

        try:
            resultado = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.ruta_base)
            )

            if resultado.returncode != 0:
                raise RuntimeError(f"Error en el escalado: {resultado.stderr}")

            print(f"Guardado: {ruta_salida}")
            return str(ruta_salida)

        except Exception as e:
            raise RuntimeError(f"Error ejecutando realesrgan: {e}")

    def escalar_carpeta(
        self,
        carpeta_entrada: str,
        carpeta_salida: Optional[str] = None,
        escala: Optional[int] = None,
        formato_salida: str = "png",
        extensiones: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ) -> list:
        """
        Escala todas las imágenes en una carpeta.

        Args:
            carpeta_entrada: Carpeta con imágenes
            carpeta_salida: Carpeta de salida (opcional)
            escala: Factor de escala
            formato_salida: Formato de salida
            extensiones: Extensiones a procesar

        Returns:
            Lista de archivos procesados
        """
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
            print("No se encontraron imágenes")
            return []

        print(f"Procesando {len(imagenes)} imágenes...")
        resultados = []

        for i, img in enumerate(imagenes, 1):
            print(f"[{i}/{len(imagenes)}] ", end="")
            try:
                nombre_salida = f"{img.stem}_x{escala or self.escala}.{formato_salida}"
                ruta_salida = carpeta_salida / nombre_salida
                self.escalar_imagen(str(img), str(ruta_salida), escala, formato_salida)
                resultados.append(str(ruta_salida))
            except Exception as e:
                print(f"Error: {e}")

        print(f"\nCompletado: {len(resultados)}/{len(imagenes)} imágenes")
        return resultados


def listar_modelos():
    """Muestra los modelos disponibles"""
    print("\n=== Modelos disponibles ===\n")
    for nombre, info in MODELOS_DISPONIBLES.items():
        print(f"  {nombre}")
        print(f"    Escala: {info['escala']}x")
        print(f"    {info['descripcion']}")
        print()


if __name__ == "__main__":
    listar_modelos()
