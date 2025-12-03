@echo off
echo ============================================
echo Instalador - Upscaler de Alta Fidelidad
echo ============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado.
    echo Descargalo de: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Crear entorno virtual
echo Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado
echo.

REM Activar entorno
call venv\Scripts\activate.bat

REM Actualizar pip
echo Actualizando pip...
python -m pip install --upgrade pip

REM Instalar PyTorch con CUDA
echo.
echo ============================================
echo Instalando PyTorch con soporte CUDA...
echo (Esto puede tardar varios minutos)
echo ============================================
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Instalar dependencias
echo.
echo Instalando dependencias adicionales...
pip install -r requirements.txt

echo.
echo ============================================
echo Instalacion completada!
echo ============================================
echo.
echo Para usar el upscaler:
echo   1. Abre una terminal en esta carpeta
echo   2. Ejecuta: venv\Scripts\activate
echo   3. Ejecuta: python upscaler.py -i imagen.jpg -m hat
echo.
pause
