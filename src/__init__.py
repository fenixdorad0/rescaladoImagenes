"""
Escalador de Imágenes con Real-ESRGAN
Módulo principal para escalado de imágenes usando modelos de IA actualizados
"""

from .upscaler import EscaladorImagenes, MODELOS_DISPONIBLES, listar_modelos

__version__ = "2.0.0"
__author__ = "Escalador de Imágenes"

__all__ = [
    "EscaladorImagenes",
    "MODELOS_DISPONIBLES",
    "listar_modelos",
]
