#!/usr/bin/env python3
"""
Upscaler de Alta Fidelidad - Versión Python
Soporta múltiples modelos de última generación:
- Real-ESRGAN (x2, x4)
- SwinIR (x2, x4)
- HAT (x2, x4) - Estado del arte
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def get_device():
    """Detecta el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Usando Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Usando CPU (será más lento)")
    return device


class RealESRGANUpscaler:
    """Upscaler usando Real-ESRGAN."""

    def __init__(self, scale=4, model_name='RealESRGAN_x4plus', device=None):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self.scale = scale
        self.device = device or get_device()

        # Configurar modelo según escala y tipo
        if model_name == 'RealESRGAN_x4plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        elif model_name == 'RealESRGAN_x4plus_anime':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        elif model_name == 'RealESRGAN_x2plus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        # Descargar modelo si no existe
        model_path = Path(__file__).parent / 'weights' / f'{model_name}.pth'
        model_path.parent.mkdir(exist_ok=True)

        if not model_path.exists():
            print(f"Descargando modelo {model_name}...")
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print("Descarga completada.")

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=512,  # Procesar en tiles para ahorrar VRAM
            tile_pad=10,
            pre_pad=0,
            half=True if self.device.type == 'cuda' else False,
            device=self.device
        )

        self.name = f"Real-ESRGAN ({model_name})"

    def upscale(self, img):
        """Escala una imagen."""
        output, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output


class SwinIRUpscaler:
    """Upscaler usando SwinIR - Excelente fidelidad."""

    def __init__(self, scale=4, device=None):
        self.scale = scale
        self.device = device or get_device()

        # Importar SwinIR
        from models.swinir import SwinIR

        # Configuración del modelo
        if scale == 2:
            model_url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x2_GAN.pth'
            model_name = 'SwinIR_x2'
        else:  # scale == 4
            model_url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
            model_name = 'SwinIR_x4'

        # Crear modelo
        self.model = SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='3conv'
        )

        # Descargar pesos
        model_path = Path(__file__).parent / 'weights' / f'{model_name}.pth'
        model_path.parent.mkdir(exist_ok=True)

        if not model_path.exists():
            print(f"Descargando modelo {model_name}...")
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print("Descarga completada.")

        # Cargar pesos
        pretrained_model = torch.load(model_path, map_location=self.device)
        if 'params_ema' in pretrained_model:
            self.model.load_state_dict(pretrained_model['params_ema'], strict=True)
        elif 'params' in pretrained_model:
            self.model.load_state_dict(pretrained_model['params'], strict=True)
        else:
            self.model.load_state_dict(pretrained_model, strict=True)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.name = f"SwinIR (x{scale})"

    def upscale(self, img):
        """Escala una imagen."""
        # Convertir BGR a RGB y normalizar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Padding para que sea divisible por window_size
        window_size = 8
        _, _, h, w = img_tensor.size()
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = self.model(img_tensor)

        # Quitar padding
        output = output[:, :, :h*self.scale, :w*self.scale]

        # Convertir a numpy
        output = output.squeeze(0).cpu().numpy()
        output = np.clip(output * 255., 0, 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output


class HATUpscaler:
    """Upscaler usando HAT - Estado del Arte en fidelidad."""

    def __init__(self, scale=4, device=None):
        self.scale = scale
        self.device = device or get_device()

        from models.hat import HAT

        # Configuración HAT-L (Large) para máxima calidad
        self.model = HAT(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

        model_name = f'HAT_x{scale}'
        model_path = Path(__file__).parent / 'weights' / f'{model_name}.pth'

        if model_path.exists():
            pretrained_model = torch.load(model_path, map_location=self.device)
            if 'params_ema' in pretrained_model:
                self.model.load_state_dict(pretrained_model['params_ema'], strict=True)
            elif 'params' in pretrained_model:
                self.model.load_state_dict(pretrained_model['params'], strict=True)
            else:
                self.model.load_state_dict(pretrained_model, strict=True)
        else:
            print(f"AVISO: No se encontró {model_path}")
            print("Descarga el modelo de: https://github.com/XPixelGroup/HAT")
            print("Y colócalo en la carpeta 'weights'")

        self.model.eval()
        self.model = self.model.to(self.device)

        self.name = f"HAT (x{scale}) - Estado del Arte"

    def upscale(self, img):
        """Escala una imagen con tiles para ahorrar memoria."""
        # Procesar en tiles para evitar OOM
        tile_size = 256
        tile_pad = 32

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Crear imagen de salida
        output = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)

        # Procesar por tiles
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Calcular bordes con padding
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x2 = min(w, x + tile_size + tile_pad)

                # Extraer tile
                tile = img_rgb[y1:y2, x1:x2]

                # Procesar tile
                tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float() / 255.
                tile_tensor = tile_tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    with torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext():
                        tile_output = self.model(tile_tensor)

                tile_output = tile_output.squeeze(0).cpu().numpy()
                tile_output = np.clip(tile_output * 255., 0, 255).astype(np.uint8)
                tile_output = tile_output.transpose(1, 2, 0)

                # Calcular posición en salida (sin padding)
                out_y1 = (y - y1) * self.scale
                out_x1 = (x - x1) * self.scale
                out_y2 = out_y1 + min(tile_size, h - y) * self.scale
                out_x2 = out_x1 + min(tile_size, w - x) * self.scale

                # Posición en imagen de salida
                dest_y1 = y * self.scale
                dest_x1 = x * self.scale
                dest_y2 = min((y + tile_size) * self.scale, h * self.scale)
                dest_x2 = min((x + tile_size) * self.scale, w * self.scale)

                output[dest_y1:dest_y2, dest_x1:dest_x2] = tile_output[out_y1:out_y2, out_x1:out_x2]

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output


def get_upscaler(model_type, scale, device):
    """Obtiene el upscaler según el tipo especificado."""
    model_type = model_type.lower()

    if model_type == 'realesrgan':
        if scale == 2:
            return RealESRGANUpscaler(scale=2, model_name='RealESRGAN_x2plus', device=device)
        else:
            return RealESRGANUpscaler(scale=4, model_name='RealESRGAN_x4plus', device=device)

    elif model_type == 'realesrgan_anime':
        return RealESRGANUpscaler(scale=4, model_name='RealESRGAN_x4plus_anime', device=device)

    elif model_type == 'swinir':
        return SwinIRUpscaler(scale=scale, device=device)

    elif model_type == 'hat':
        return HATUpscaler(scale=scale, device=device)

    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")


def process_image(input_path, output_path, upscaler, output_format='png'):
    """Procesa una imagen individual."""
    print(f"\nProcesando: {input_path}")
    start_time = time.time()

    # Leer imagen
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: No se pudo leer {input_path}")
        return False

    h, w = img.shape[:2]
    print(f"Tamaño original: {w}x{h}")

    # Procesar
    try:
        output = upscaler.upscale(img)
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return False

    oh, ow = output.shape[:2]
    print(f"Tamaño final: {ow}x{oh}")

    # Guardar
    output_path = Path(output_path)
    if output_format.lower() == 'jpg':
        cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    elapsed = time.time() - start_time
    print(f"Tiempo: {elapsed:.2f}s")
    print(f"Guardado en: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Upscaler de Alta Fidelidad - Modelos de Última Generación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modelos disponibles:
  realesrgan       - Real-ESRGAN x4plus (buena calidad general)
  realesrgan_anime - Real-ESRGAN optimizado para anime
  swinir           - SwinIR (excelente fidelidad, más lento)
  hat              - HAT (estado del arte, máxima fidelidad)

Ejemplos:
  python upscaler.py -i imagen.jpg -m realesrgan
  python upscaler.py -i carpeta/ -m hat -s 4 -f png
  python upscaler.py -i foto.png -m swinir -s 2 -o foto_2x.png
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Imagen o carpeta de entrada')
    parser.add_argument('-o', '--output', default=None,
                        help='Imagen o carpeta de salida (opcional)')
    parser.add_argument('-m', '--model', default='realesrgan',
                        choices=['realesrgan', 'realesrgan_anime', 'swinir', 'hat'],
                        help='Modelo a usar (default: realesrgan)')
    parser.add_argument('-s', '--scale', type=int, default=4, choices=[2, 4],
                        help='Factor de escala (default: 4)')
    parser.add_argument('-f', '--format', default='png', choices=['png', 'jpg'],
                        help='Formato de salida (default: png)')
    parser.add_argument('--cpu', action='store_true',
                        help='Forzar uso de CPU')

    args = parser.parse_args()

    # Configurar dispositivo
    if args.cpu:
        device = torch.device('cpu')
        print("Forzando uso de CPU")
    else:
        device = get_device()

    # Inicializar upscaler
    print(f"\nCargando modelo: {args.model} (x{args.scale})...")
    upscaler = get_upscaler(args.model, args.scale, device)
    print(f"Modelo cargado: {upscaler.name}")

    input_path = Path(args.input)

    # Procesar archivo o carpeta
    if input_path.is_file():
        # Archivo individual
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_{args.model}_x{args.scale}.{args.format}"

        success = process_image(input_path, output_path, upscaler, args.format)

    elif input_path.is_dir():
        # Carpeta
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path / 'upscaled'
        output_dir.mkdir(exist_ok=True)

        # Buscar imágenes
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        images = [f for f in input_path.iterdir()
                  if f.suffix.lower() in extensions]

        print(f"\nEncontradas {len(images)} imágenes")

        success_count = 0
        for img_path in images:
            output_path = output_dir / f"{img_path.stem}_x{args.scale}.{args.format}"
            if process_image(img_path, output_path, upscaler, args.format):
                success_count += 1

        print(f"\n{'='*50}")
        print(f"Procesadas: {success_count}/{len(images)} imágenes")
        print(f"Guardadas en: {output_dir}")
    else:
        print(f"Error: {input_path} no existe")
        sys.exit(1)

    print("\n¡Completado!")


# Contexto nulo para cuando no hay CUDA
class nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): pass


if __name__ == '__main__':
    main()
