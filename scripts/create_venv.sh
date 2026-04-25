echo "Virtual env created at $VENV_PATH"
#!/usr/bin/env bash
# create_venv.sh [<venv_path>]
# Creates a venv (default: .uv) and upgrades pip/setuptools/wheel.
set -euo pipefail
VENV_PATH=".uv"
if [ "$#" -ge 1 ]; then
  VENV_PATH="$1"
fi
python3 -m venv "$VENV_PATH"
# Use the venv's python to upgrade pip/setuptools/wheel (avoid global pip)
 "$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
echo "Virtual env created at $VENV_PATH"
