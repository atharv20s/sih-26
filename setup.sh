#!/usr/bin/env bash
# PRAHARI - one-command bootstrap (macOS / Linux / WSL / Git Bash)
# See setup.ps1 for the Windows PowerShell equivalent - same steps.
#
# Usage:
#   ./setup.sh                # full bootstrap + launch
#   ./setup.sh --no-launch    # bootstrap only, don't start the server
#   ./setup.sh --skip-db      # skip DB table creation / user seeding

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NO_LAUNCH=0
SKIP_DB=0
for arg in "$@"; do
  case "$arg" in
    --no-launch) NO_LAUNCH=1 ;;
    --skip-db) SKIP_DB=1 ;;
  esac
done

step() { echo ""; echo "==> $1"; }

step "Checking Python"
PYCMD="python3"
command -v python3 >/dev/null 2>&1 || PYCMD="python"
if ! command -v "$PYCMD" >/dev/null 2>&1; then
  echo "Python not found on PATH. Install Python 3.10+ and re-run."
  exit 1
fi
"$PYCMD" --version

step "Creating virtual environment (.venv) if needed"
if [ ! -d ".venv" ]; then
  "$PYCMD" -m venv .venv
  echo "  Created .venv"
else
  echo "  .venv already exists"
fi
VENV_PY=".venv/bin/python"
[ -f "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"  # Git Bash on Windows

step "Installing dependencies (this can take a few minutes on first run)"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r requirements.txt
echo "  Done"

step "Checking .env"
if [ ! -f ".env" ]; then
  cp .env.example .env
  JWT_SECRET=$("$VENV_PY" -c "import secrets; print(secrets.token_hex(32))")
  HMAC_SECRET=$("$VENV_PY" -c "import secrets; print(secrets.token_hex(24))")
  # portable in-place sed (no -i suffix quirk between BSD/GNU)
  "$VENV_PY" - "$JWT_SECRET" "$HMAC_SECRET" <<'PYEOF'
import sys
jwt_secret, hmac_secret = sys.argv[1], sys.argv[2]
with open(".env") as f:
    content = f.read()
content = content.replace("JWT_SECRET=change-me", f"JWT_SECRET={jwt_secret}")
content = content.replace("TELEMETRY_HMAC_SECRET=change-me", f"TELEMETRY_HMAC_SECRET={hmac_secret}")
with open(".env", "w") as f:
    f.write(content)
PYEOF
  echo "  Created .env with generated JWT/HMAC secrets."
  echo "  ACTION NEEDED: open .env and set DATABASE_URL (Postgres/Neon connection string)."
  echo "  Optional: set GLM_API_KEY to enable the AI Advisor panel."
else
  echo "  .env already exists - leaving it as-is"
fi

HAS_REAL_DB=0
if grep -q "^DATABASE_URL=" .env && ! grep -q "user:password@localhost" .env; then
  HAS_REAL_DB=1
fi

if [ "$SKIP_DB" -eq 0 ]; then
  if [ "$HAS_REAL_DB" -eq 1 ]; then
    step "Creating database tables"
    "$VENV_PY" scripts/init_db.py

    step "Checking for a seeded demo login"
    HAS_USER=$("$VENV_PY" scripts/check_login_exists.py)
    if [ "$HAS_USER" = "no" ]; then
      GEN_PASSWORD=$("$VENV_PY" -c "import secrets; print(secrets.token_urlsafe(9))")
      "$VENV_PY" scripts/create_user.py --email judge@prahari.app --password "$GEN_PASSWORD" --name "SIH Judge"
      echo ""
      echo "  Seeded a login - SAVE THIS PASSWORD, it will not be shown again:"
      echo "    email:    judge@prahari.app"
      echo "    password: $GEN_PASSWORD"
    else
      echo "  A login already exists - skipping seed (run scripts/create_user.py yourself to add another)."
    fi
  else
    echo ""
    echo "  Skipping database setup - DATABASE_URL in .env still looks like the placeholder."
    echo "  Set a real Postgres connection string, then re-run: ./setup.sh --no-launch"
  fi
fi

echo ""
echo "==> Bootstrap complete."

if [ "$NO_LAUNCH" -eq 0 ]; then
  step "Launching PRAHARI server at http://localhost:8000"
  "$VENV_PY" -m uvicorn server.server:app --app-dir src --host 0.0.0.0 --port 8000
else
  echo "  Run this to launch later:"
  echo "    $VENV_PY -m uvicorn server.server:app --app-dir src --host 0.0.0.0 --port 8000"
fi
