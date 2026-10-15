# ANATO3D — run the full test suite (backend + frontend).
#
#   pwsh -File scripts/run_tests.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

Write-Host "==> Backend tests (pytest)" -ForegroundColor Cyan
Push-Location $backend
python -m pytest -q
$backendExit = $LASTEXITCODE
Pop-Location

Write-Host "==> Frontend typecheck" -ForegroundColor Cyan
Push-Location $frontend
npm run typecheck
$typecheckExit = $LASTEXITCODE

Write-Host "==> Frontend tests (vitest)" -ForegroundColor Cyan
npm test
$frontendExit = $LASTEXITCODE
Pop-Location

Write-Host ""
if ($backendExit -eq 0 -and $typecheckExit -eq 0 -and $frontendExit -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
    exit 0
}

Write-Host "One or more checks failed." -ForegroundColor Red
exit 1
