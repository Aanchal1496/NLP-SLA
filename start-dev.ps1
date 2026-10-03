# Development startup: backend API + bundled dashboard (single service).
# The FastAPI app serves the frontend at /app, so one process is enough.
# Usage: powershell -ExecutionPolicy Bypass -File start-dev.ps1 [-Port 8000]
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path "app/main.py")) {
    Write-Error "Run this script from the project root (folder containing app/)."
    exit 1
}

try {
    python --version | Out-Null
} catch {
    Write-Error "Python 3.12 is required but was not found on PATH."
    exit 1
}

Write-Output "Installing backend dependencies (pinned)..."
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed -- see pip output above."
    exit 1
}

Write-Output ""
Write-Output "Starting backend + dashboard on http://127.0.0.1:${Port} ..."
Write-Output "  Website : http://127.0.0.1:${Port}/app/"
Write-Output "  API docs: http://127.0.0.1:${Port}/docs"
Write-Output "Press Ctrl+C to stop."
uvicorn app.main:app --host 127.0.0.1 --port $Port
