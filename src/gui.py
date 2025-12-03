#!/usr/bin/env python3
"""
Interfaz gráfica para el escalador de imágenes con Real-ESRGAN
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import queue
from typing import Optional


class EscaladorGUI:
    """Interfaz gráfica para el escalador de imágenes"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Escalador de Imágenes - Real-ESRGAN")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Cola para comunicación entre hilos
        self.cola_mensajes = queue.Queue()

        # Variables
        self.ruta_entrada = tk.StringVar()
        self.ruta_salida = tk.StringVar()
        self.modelo_seleccionado = tk.StringVar(value="realesrgan-x4plus")
        self.escala_seleccionada = tk.StringVar(value="4")
        self.formato_seleccionado = tk.StringVar(value="png")
        self.gpu_id = tk.IntVar(value=0)
        self.usar_tile = tk.BooleanVar(value=False)
        self.procesando = False
        self.escalador = None

        self._crear_interfaz()
        self._iniciar_verificador_cola()

    def _crear_interfaz(self):
        """Crea todos los elementos de la interfaz"""
        # Frame principal con padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        fila = 0

        # === Sección de entrada ===
        ttk.Label(main_frame, text="Imagen o Carpeta de Entrada:",
                  font=("", 10, "bold")).grid(row=fila, column=0, columnspan=3, sticky="w", pady=(0, 5))
        fila += 1

        ttk.Entry(main_frame, textvariable=self.ruta_entrada, width=50).grid(
            row=fila, column=0, columnspan=2, sticky="ew", padx=(0, 5))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=fila, column=2, sticky="e")
        ttk.Button(btn_frame, text="Archivo", command=self._seleccionar_archivo).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Carpeta", command=self._seleccionar_carpeta).pack(side="left", padx=2)
        fila += 1

        # === Sección de salida ===
        ttk.Label(main_frame, text="Carpeta de Salida (opcional):",
                  font=("", 10, "bold")).grid(row=fila, column=0, columnspan=3, sticky="w", pady=(15, 5))
        fila += 1

        ttk.Entry(main_frame, textvariable=self.ruta_salida, width=50).grid(
            row=fila, column=0, columnspan=2, sticky="ew", padx=(0, 5))
        ttk.Button(main_frame, text="Seleccionar", command=self._seleccionar_salida).grid(
            row=fila, column=2, sticky="e")
        fila += 1

        # === Opciones ===
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=fila, column=0, columnspan=3, sticky="ew", pady=15)
        fila += 1

        ttk.Label(main_frame, text="Opciones:", font=("", 10, "bold")).grid(
            row=fila, column=0, columnspan=3, sticky="w", pady=(0, 10))
        fila += 1

        # Frame para opciones
        opciones_frame = ttk.Frame(main_frame)
        opciones_frame.grid(row=fila, column=0, columnspan=3, sticky="ew")
        fila += 1

        # Modelo
        ttk.Label(opciones_frame, text="Modelo:").grid(row=0, column=0, sticky="w", pady=5)
        modelos = [
            ("realesrgan-x4plus", "General x4 (Alta calidad)"),
            ("realesrgan-x4plus-anime", "Anime x4"),
            ("realesrgan-x2plus", "General x2 (Rápido)"),
            ("realesr-general-x4v3", "General v3 (Última versión)"),
        ]
        modelo_combo = ttk.Combobox(opciones_frame, textvariable=self.modelo_seleccionado,
                                     values=[m[0] for m in modelos], state="readonly", width=30)
        modelo_combo.grid(row=0, column=1, sticky="w", padx=10)

        # Escala
        ttk.Label(opciones_frame, text="Escala:").grid(row=1, column=0, sticky="w", pady=5)
        escala_frame = ttk.Frame(opciones_frame)
        escala_frame.grid(row=1, column=1, sticky="w", padx=10)
        for escala in ["2", "4", "8", "16"]:
            ttk.Radiobutton(escala_frame, text=f"{escala}x", variable=self.escala_seleccionada,
                           value=escala).pack(side="left", padx=5)

        # Formato
        ttk.Label(opciones_frame, text="Formato:").grid(row=2, column=0, sticky="w", pady=5)
        formato_frame = ttk.Frame(opciones_frame)
        formato_frame.grid(row=2, column=1, sticky="w", padx=10)
        for fmt in ["png", "jpg", "webp"]:
            ttk.Radiobutton(formato_frame, text=fmt.upper(), variable=self.formato_seleccionado,
                           value=fmt).pack(side="left", padx=5)

        # GPU
        ttk.Label(opciones_frame, text="GPU:").grid(row=3, column=0, sticky="w", pady=5)
        gpu_frame = ttk.Frame(opciones_frame)
        gpu_frame.grid(row=3, column=1, sticky="w", padx=10)
        ttk.Spinbox(gpu_frame, from_=-1, to=7, textvariable=self.gpu_id, width=5).pack(side="left")
        ttk.Label(gpu_frame, text="(-1 para CPU)").pack(side="left", padx=10)

        # Tile (para imágenes grandes)
        ttk.Checkbutton(opciones_frame, text="Usar tiles (para imágenes muy grandes, usa menos VRAM)",
                        variable=self.usar_tile).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        # === Barra de progreso ===
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=fila, column=0, columnspan=3, sticky="ew", pady=15)
        fila += 1

        self.barra_progreso = ttk.Progressbar(main_frame, mode="indeterminate", length=400)
        self.barra_progreso.grid(row=fila, column=0, columnspan=3, sticky="ew", pady=5)
        fila += 1

        self.etiqueta_estado = ttk.Label(main_frame, text="Listo para procesar", foreground="gray")
        self.etiqueta_estado.grid(row=fila, column=0, columnspan=3, sticky="w")
        fila += 1

        # === Botones de acción ===
        botones_frame = ttk.Frame(main_frame)
        botones_frame.grid(row=fila, column=0, columnspan=3, pady=20)

        self.btn_procesar = ttk.Button(botones_frame, text="Escalar Imagen",
                                       command=self._iniciar_procesamiento)
        self.btn_procesar.pack(side="left", padx=10)

        ttk.Button(botones_frame, text="Limpiar", command=self._limpiar).pack(side="left", padx=10)

    def _seleccionar_archivo(self):
        """Abre diálogo para seleccionar archivo"""
        archivo = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if archivo:
            self.ruta_entrada.set(archivo)

    def _seleccionar_carpeta(self):
        """Abre diálogo para seleccionar carpeta de entrada"""
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta con imágenes")
        if carpeta:
            self.ruta_entrada.set(carpeta)

    def _seleccionar_salida(self):
        """Abre diálogo para seleccionar carpeta de salida"""
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if carpeta:
            self.ruta_salida.set(carpeta)

    def _limpiar(self):
        """Limpia los campos"""
        self.ruta_entrada.set("")
        self.ruta_salida.set("")
        self.etiqueta_estado.config(text="Listo para procesar", foreground="gray")

    def _actualizar_estado(self, mensaje: str, color: str = "black"):
        """Actualiza la etiqueta de estado"""
        self.etiqueta_estado.config(text=mensaje, foreground=color)

    def _iniciar_verificador_cola(self):
        """Inicia el verificador periódico de la cola de mensajes"""
        try:
            while True:
                mensaje, color = self.cola_mensajes.get_nowait()
                self._actualizar_estado(mensaje, color)
        except queue.Empty:
            pass
        self.root.after(100, self._iniciar_verificador_cola)

    def _iniciar_procesamiento(self):
        """Inicia el procesamiento en un hilo separado"""
        if self.procesando:
            return

        entrada = self.ruta_entrada.get().strip()
        if not entrada:
            messagebox.showerror("Error", "Selecciona una imagen o carpeta")
            return

        if not Path(entrada).exists():
            messagebox.showerror("Error", f"No se encontró: {entrada}")
            return

        self.procesando = True
        self.btn_procesar.config(state="disabled")
        self.barra_progreso.start(10)

        # Ejecutar en hilo separado
        hilo = threading.Thread(target=self._procesar, daemon=True)
        hilo.start()

    def _procesar(self):
        """Realiza el procesamiento (ejecutado en hilo separado)"""
        try:
            from upscaler import EscaladorImagenes

            entrada = Path(self.ruta_entrada.get().strip())
            salida = self.ruta_salida.get().strip() or None
            modelo = self.modelo_seleccionado.get()
            escala = int(self.escala_seleccionada.get())
            formato = self.formato_seleccionado.get()
            gpu = self.gpu_id.get()
            tile = 400 if self.usar_tile.get() else 0

            self.cola_mensajes.put(("Cargando modelo...", "blue"))

            # Inicializar escalador
            escalador = EscaladorImagenes(
                modelo=modelo,
                gpu_id=gpu,
                tile=tile
            )

            self.cola_mensajes.put(("Procesando...", "blue"))

            if entrada.is_file():
                resultado = escalador.escalar_imagen(
                    str(entrada), salida, escala, formato
                )
                self.cola_mensajes.put((f"¡Completado! Guardado en: {resultado}", "green"))
            else:
                resultados = escalador.escalar_carpeta(
                    str(entrada), salida, escala, formato
                )
                self.cola_mensajes.put((f"¡Completado! {len(resultados)} imágenes procesadas", "green"))

            self.root.after(0, lambda: messagebox.showinfo("Éxito", "Procesamiento completado"))

        except Exception as e:
            self.cola_mensajes.put((f"Error: {str(e)}", "red"))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.root.after(0, self._finalizar_procesamiento)

    def _finalizar_procesamiento(self):
        """Finaliza el procesamiento y restaura la interfaz"""
        self.procesando = False
        self.btn_procesar.config(state="normal")
        self.barra_progreso.stop()

    def ejecutar(self):
        """Inicia la aplicación"""
        self.root.mainloop()


def main():
    app = EscaladorGUI()
    app.ejecutar()


if __name__ == "__main__":
    main()
