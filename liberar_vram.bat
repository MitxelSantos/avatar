@echo off
echo ========================================
echo LIBERANDO VRAM PARA ENTRENAMIENTO
echo ========================================
echo.

echo Cerrando Chrome...
taskkill /F /IM chrome.exe 2>nul

echo Cerrando VS Code...
taskkill /F /IM Code.exe 2>nul

echo Cerrando WhatsApp...
taskkill /F /IM WhatsApp.exe 2>nul

echo Cerrando Steam Helper...
taskkill /F /IM steamwebhelper.exe 2>nul

echo Cerrando OneDrive...
taskkill /F /IM OneDrive.exe 2>nul

echo Cerrando Video UI...
taskkill /F /IM Video.UI.exe 2>nul

echo.
echo Esperando 5 segundos para liberar recursos...
timeout /t 5 /nobreak

echo.
echo ========================================
echo VERIFICANDO VRAM DISPONIBLE
echo ========================================
nvidia-smi

echo.
echo ========================================
echo Si VRAM usado es ^< 200MB, presiona Enter
echo para iniciar entrenamiento
echo ========================================
pause