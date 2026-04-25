Contribution guidelines

- Use branches for features or experiments.
- Add tests for new utilities and keep notebooks for exploration.
- Keep `main/` for reproducible experiment runners.

Run tests (use the repo virtual env `.uv`):

```bash
bash scripts/create_venv.sh  # creates .uv if missing
source .uv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```
