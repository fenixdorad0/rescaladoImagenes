#!/usr/bin/env python3
"""
CLI para el escalador de imágenes con Real-ESRGAN
Uso: python cli.py [opciones] imagen.jpg
"""

import argparse
import sys
from pathlib import Path

from upscaler import EscaladorImagenes, MODELOS_DISPONIBLES, listar_modelos


def crear_parser():
    """Crea el parser de argumentos"""
    parser = argparse.ArgumentParser(
        description="Escala imágenes usando Real-ESRGAN con modelos de IA actualizados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python cli.py imagen.jpg                    # Escalar con modelo por defecto (x4)
  python cli.py imagen.jpg -m anime           # Usar modelo optimizado para anime
  python cli.py imagen.jpg -s 8               # Escalar a 8x (aplica 4x dos veces)
  python cli.py carpeta/ -o salida/           # Escalar toda una carpeta
  python cli.py imagen.jpg --listar-modelos   # Ver modelos disponibles

Modelos disponibles:
  realesrgan-x4plus       - Fotos reales, alta calidad (x4)
  realesrgan-x4plus-anime - Anime e ilustraciones (x4)
  realesrgan-x2plus       - Fotos reales, más rápido (x2)
  realesr-general-x4v3    - Última versión, mejor calidad (x4)
        """
    )

    parser.add_argument(
        "entrada",
        nargs="?",
        help="Imagen o carpeta de entrada"
    )

    parser.add_argument(
        "-o", "--salida",
        help="Ruta de salida (archivo o carpeta)"
    )

    parser.add_argument(
        "-m", "--modelo",
        default="realesrgan-x4plus",
        choices=list(MODELOS_DISPONIBLES.keys()) + ["anime", "general", "rapido"],
        help="Modelo a usar (default: realesrgan-x4plus)"
    )

    parser.add_argument(
        "-s", "--escala",
        type=int,
        choices=[2, 4, 8, 16],
        help="Escala objetivo (2, 4, 8, 16)"
    )

    parser.add_argument(
        "-f", "--formato",
        default="png",
        choices=["png", "jpg", "webp"],
        help="Formato de salida (default: png)"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="ID de GPU a usar (default: 0, usa -1 para CPU)"
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tamaño de tile para imágenes grandes (0=desactivado)"
    )

    parser.add_argument(
        "--listar-modelos",
        action="store_true",
        help="Muestra los modelos disponibles"
    )

    parser.add_argument(
        "--sin-fp16",
        action="store_true",
        help="Desactiva precisión media (usa más VRAM)"
    )

    return parser


def resolver_alias_modelo(modelo: str) -> str:
    """Resuelve alias de modelos a nombres completos"""
    aliases = {
        "anime": "realesrgan-x4plus-anime",
        "general": "realesr-general-x4v3",
        "rapido": "realesrgan-x2plus",
    }
    return aliases.get(modelo, modelo)


def main():
    parser = crear_parser()
    args = parser.parse_args()

    # Listar modelos si se solicita
    if args.listar_modelos:
        listar_modelos()
        return 0

    # Validar entrada
    if not args.entrada:
        parser.print_help()
        print("\nError: Debes especificar una imagen o carpeta de entrada")
        return 1

    entrada = Path(args.entrada)
    if not entrada.exists():
        print(f"Error: No se encontró '{entrada}'")
        return 1

    # Resolver alias de modelo
    modelo = resolver_alias_modelo(args.modelo)

    try:
        # Inicializar escalador
        print(f"\nInicializando Real-ESRGAN...")
        escalador = EscaladorImagenes(
            modelo=modelo,
            gpu_id=args.gpu,
            tile=args.tile,
            half_precision=not args.sin_fp16
        )

        # Procesar
        if entrada.is_file():
            # Escalar una imagen
            print(f"\nProcesando: {entrada.name}")
            resultado = escalador.escalar_imagen(
                str(entrada),
                args.salida,
                args.escala,
                args.formato
            )
            print(f"\n¡Completado! Imagen guardada en: {resultado}")

        elif entrada.is_dir():
            # Escalar carpeta
            print(f"\nProcesando carpeta: {entrada}")
            resultados = escalador.escalar_carpeta(
                str(entrada),
                args.salida,
                args.escala,
                args.formato
            )
            print(f"\n¡Completado! {len(resultados)} imágenes procesadas")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
