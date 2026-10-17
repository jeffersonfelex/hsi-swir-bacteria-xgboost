# Script de execução rápida para PowerShell (IC-2025)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Warning "Ambiente .venv não encontrado. Usando python do sistema..."
    $PythonExe = "python"
}

& $PythonExe (Join-Path $ScriptDir "run.py") @args
