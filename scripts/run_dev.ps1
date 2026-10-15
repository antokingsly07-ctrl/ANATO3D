# ANATO3D — start the backend and frontend for local development (Windows).
#
#   pwsh -File scripts/run_dev.ps1
#
# Backend  -> http://127.0.0.1:8000
# Frontend -> http://127.0.0.1:5173  (Vite dev server, proxies /api to backend)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

function Assert-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "'$name' was not found on PATH. $hint"
    }
}

Assert-Command "python" "Install Python 3.10+ and make sure it is on PATH."
Assert-Command "npm" "Install Node.js 18+ and make sure it is on PATH."

Write-Host "Starting ANATO3D backend (http://127.0.0.1:8000) ..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "run.py" -WorkingDirectory $backend

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies (first run) ..." -ForegroundColor Yellow
    Push-Location $frontend
    npm install
    Pop-Location
}

Write-Host "Starting ANATO3D frontend (http://127.0.0.1:5173) ..." -ForegroundColor Cyan
Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory $frontend

Write-Host ""
Write-Host "Both processes started in separate windows." -ForegroundColor Green
Write-Host "  Backend : http://127.0.0.1:8000/docs"
Write-Host "  Frontend: http://127.0.0.1:5173"
Write-Host "Close those windows (or stop the processes) to shut down."
