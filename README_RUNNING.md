Quick run and verification

1. Create venv (default: `.uv`):

```bash
bash scripts/create_venv.sh
source .uv/bin/activate
python -m pip install -r requirements.txt
```

2. Run a quick smoke test (no extra dependencies):

```bash
python -c "from src import __version__; print('version', __version__)"
```

3. Run tests:

```bash
python -m pytest -q
```
