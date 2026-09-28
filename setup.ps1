# PRAHARI - one-command bootstrap (Windows PowerShell)
#
# Does everything scripts/README.md previously asked a human to do by hand:
#   1. Create/activate a local virtualenv (.venv)
#   2. Install requirements.txt
#   3. Create .env from .env.example if missing, auto-generating the two
#      secrets that don't need an external account (JWT_SECRET,
#      TELEMETRY_HMAC_SECRET) - DATABASE_URL and GLM_API_KEY still need a
#      real value pasted in by hand (they require external accounts).
#   4. Create Postgres tables (skipped gracefully if DATABASE_URL isn't set yet)
#   5. Seed a demo login if one doesn't already exist
#   6. Launch the server
#
# Usage:
#   .\setup.ps1                 # full bootstrap + launch
#   .\setup.ps1 -NoLaunch       # bootstrap only, don't start the server
#   .\setup.ps1 -SkipDb         # skip DB table creation / user seeding

param(
    [switch]$NoLaunch,
    [switch]$SkipDb
)

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

Write-Step "Checking Python"
$pyCmd = "python"
try {
    $pyVersion = & $pyCmd --version 2>&1
    Write-Host "  $pyVersion"
} catch {
    Write-Host "Python not found on PATH. Install Python 3.10+ and re-run." -ForegroundColor Red
    exit 1
}

Write-Step "Creating virtual environment (.venv) if needed"
if (-not (Test-Path "$ROOT\.venv")) {
    & $pyCmd -m venv "$ROOT\.venv"
    Write-Host "  Created .venv"
} else {
    Write-Host "  .venv already exists"
}
$venvPython = "$ROOT\.venv\Scripts\python.exe"

Write-Step "Installing dependencies (this can take a few minutes on first run)"
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet -r requirements.txt
Write-Host "  Done"

Write-Step "Checking .env"
if (-not (Test-Path "$ROOT\.env")) {
    Copy-Item "$ROOT\.env.example" "$ROOT\.env"
    # Auto-generate the two secrets that don't need an external account
    $jwtSecret = & $venvPython -c "import secrets; print(secrets.token_hex(32))"
    $hmacSecret = & $venvPython -c "import secrets; print(secrets.token_hex(24))"
    (Get-Content "$ROOT\.env") `
        -replace "JWT_SECRET=change-me", "JWT_SECRET=$jwtSecret" `
        -replace "TELEMETRY_HMAC_SECRET=change-me", "TELEMETRY_HMAC_SECRET=$hmacSecret" `
        | Set-Content "$ROOT\.env"
    Write-Host "  Created .env with generated JWT/HMAC secrets." -ForegroundColor Yellow
    Write-Host "  ACTION NEEDED: open .env and set DATABASE_URL (Postgres/Neon connection string)." -ForegroundColor Yellow
    Write-Host "  Optional: set GLM_API_KEY to enable the AI Advisor panel." -ForegroundColor Yellow
} else {
    Write-Host "  .env already exists - leaving it as-is"
}

# Load .env into this process so we can check whether DATABASE_URL looks real
$envMap = @{}
Get-Content "$ROOT\.env" | ForEach-Object {
    if ($_ -match "^\s*([A-Z_]+)\s*=\s*(.*)$") {
        $envMap[$matches[1]] = $matches[2]
    }
}
$hasRealDb = $envMap.ContainsKey("DATABASE_URL") -and $envMap["DATABASE_URL"] -notmatch "user:password@localhost"

if (-not $SkipDb) {
    if ($hasRealDb) {
        Write-Step "Creating database tables"
        & $venvPython scripts\init_db.py

        Write-Step "Checking for a seeded demo login"
        $checkUser = (& $venvPython scripts\check_login_exists.py 2>&1 | Select-Object -Last 1).Trim()
        if ($checkUser -eq "no") {
            $genPassword = & $venvPython -c "import secrets; print(secrets.token_urlsafe(9))"
            & $venvPython scripts\create_user.py --email judge@prahari.app --password $genPassword --name "SIH Judge"
            Write-Host ""
            Write-Host "  Seeded a login - SAVE THIS PASSWORD, it will not be shown again:" -ForegroundColor Yellow
            Write-Host "    email:    judge@prahari.app" -ForegroundColor Yellow
            Write-Host "    password: $genPassword" -ForegroundColor Yellow
        } else {
            Write-Host "  A login already exists - skipping seed (run scripts\create_user.py yourself to add another)."
        }
    } else {
        Write-Host ""
        Write-Host "  Skipping database setup - DATABASE_URL in .env still looks like the placeholder." -ForegroundColor Yellow
        Write-Host "  Set a real Postgres connection string, then re-run: .\setup.ps1 -NoLaunch" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "==> Bootstrap complete." -ForegroundColor Green

if (-not $NoLaunch) {
    Write-Step "Launching PRAHARI server at http://localhost:8000"
    & $venvPython -m uvicorn server.server:app --app-dir src --host 0.0.0.0 --port 8000
} else {
    Write-Host "  Run this to launch later:"
    Write-Host "    .venv\Scripts\python.exe -m uvicorn server.server:app --app-dir src --host 0.0.0.0 --port 8000"
}
