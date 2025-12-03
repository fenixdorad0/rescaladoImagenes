#!/usr/bin/env python3
"""
Punto de entrada principal para el escalador de imágenes
Ejecuta la GUI por defecto, o CLI si se pasan argumentos
"""

import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    if len(sys.argv) > 1:
        # Si hay argumentos, usar CLI
        from cli import main as cli_main
        return cli_main()
    else:
        # Sin argumentos, abrir GUI
        from gui import main as gui_main
        gui_main()
        return 0


if __name__ == "__main__":
    sys.exit(main())
