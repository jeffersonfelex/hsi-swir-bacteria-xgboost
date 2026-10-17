@echo off
setlocal
cd /d "%~dp0"

REM Configurar codificacao UTF-8 no terminal Windows
chcp 65001 >nul

REM Usar Python do ambiente virtual .venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [AVISO] Ambiente virtual .venv nao encontrado. Usando python padrao...
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" run.py %*
endlocal
