#!/usr/bin/env python3
"""
Escalador de imágenes simple - Usa el ejecutable incluido
No requiere instalación de dependencias pesadas como PyTorch
"""

import sys
import os
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(
        description="Escalar imágenes con Real-ESRGAN (versión simple)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python escalar_simple.py imagen.jpg              # Escalar 4x
  python escalar_simple.py imagen.jpg -s 2         # Escalar 2x
  python escalar_simple.py carpeta/ -o salida/     # Escalar carpeta
  python escalar_simple.py --modelos               # Ver modelos

Modelos:
  realesrgan-x4plus        - Fotos reales (x4)
  realesrgan-x4plus-anime  - Anime (x4)
  realesr-animevideov3     - Video anime (x2,x3,x4)
        """
    )

    parser.add_argument("entrada", nargs="?", help="Imagen o carpeta")
    parser.add_argument("-o", "--salida", help="Ruta de salida")
    parser.add_argument("-m", "--modelo", default="realesrgan-x4plus",
                        help="Modelo a usar")
    parser.add_argument("-s", "--escala", type=int, choices=[2, 3, 4],
                        default=4, help="Factor de escala")
    parser.add_argument("-f", "--formato", default="png",
                        choices=["png", "jpg", "webp"], help="Formato salida")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID de GPU (-1 para CPU)")
    parser.add_argument("--modelos", action="store_true",
                        help="Listar modelos disponibles")

    args = parser.parse_args()

    from upscaler_simple import EscaladorImagenes, listar_modelos

    if args.modelos:
        listar_modelos()
        return 0

    if not args.entrada:
        parser.print_help()
        return 1

    entrada = Path(args.entrada)
    if not entrada.exists():
        print(f"Error: No existe '{entrada}'")
        return 1

    try:
        escalador = EscaladorImagenes(
            modelo=args.modelo,
            gpu_id=args.gpu
        )

        if entrada.is_file():
            resultado = escalador.escalar_imagen(
                str(entrada),
                args.salida,
                args.escala,
                args.formato
            )
            print(f"\n¡Listo! Guardado en: {resultado}")
        else:
            resultados = escalador.escalar_carpeta(
                str(entrada),
                args.salida,
                args.escala,
                args.formato
            )
            print(f"\n¡Listo! {len(resultados)} imágenes procesadas")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
