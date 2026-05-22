$ErrorActionPreference = "Stop"

Write-Host "Arabic NLP Platform teammate setup" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
  Write-Host "[INFO] Creating virtual environment..."
  py -3 -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
$pip = ".\.venv\Scripts\pip.exe"

Write-Host "[INFO] Upgrading pip..."
& $python -m pip install --upgrade pip

Write-Host "[INFO] Installing required Python packages..."
& $pip install -r requirements.txt

Write-Host "[INFO] Installing downloadable models..."
& $python install_models.py

Write-Host "[INFO] Checking Java..."
try {
  java -version 2>$null
  Write-Host "[OK] Java detected." -ForegroundColor Green
} catch {
  Write-Host "[WARN] Java not found. Farasa, AlKhalil, and MADAMIRA may be unavailable." -ForegroundColor Yellow
}

Write-Host "[INFO] Running startup validation..."
& $python startup_check.py

Write-Host ""
Write-Host "Setup complete. Start backend with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m uvicorn main:app --reload"
Write-Host "Start frontend with:"
Write-Host "  cd frontend; npm install; npm run dev"
