@echo off
echo ============================================
echo Upscaler HAT - Maxima Calidad
echo ============================================
echo.

call venv\Scripts\activate.bat

if "%~1"=="" (
    echo Uso: escalar_hat.bat "ruta/imagen.jpg"
    echo.
    echo Arrastra una imagen sobre este archivo para escalarla.
    pause
    exit /b 1
)

echo Procesando: %~1
echo Modelo: HAT (Estado del Arte)
echo Escala: 4x
echo.

python upscaler.py -i "%~1" -m hat -s 4 -f png

echo.
echo Completado!
pause
