# CNN Thesis Project

Repository skeleton for training and fine-tuning a convolutional neural network for my thesis.

Structure
- `preparatory_phase/` - experiments, data exploration, helper scripts done before main thesis work.
- `main/` - main training/fine-tuning pipelines, final experiments.
- `src/` - library code, dataset utilities, model wrappers (no CNN code included for now).
- `notebooks/` - exploratory notebooks.
- `data/` - raw and processed datasets (gitignored by default).
- `tests/` - unit tests and smoke tests.
- `scripts/` - helper scripts (venv creation, data download, etc.).

Setup
1. Create a virtual environment using the included script (default name: `.uv`):

```bash
bash scripts/create_venv.sh    # creates .uv by default
source .uv/bin/activate
python -m pip install -r requirements.txt
```

2. See `docs/` for guidance on thesis structure and experiment logging.

Notes
- This repo intentionally contains no CNN implementation yet. Add models under `src/` and training code under `main/` as you progress.

Large files and venvs

- Do NOT commit virtual environments (the `.uv/` folder is already in `.gitignore`).
- Avoid committing large model binaries (PyTorch wheel or pre-trained weights). For large artifacts consider:
	- Git Large File Storage (Git LFS) for versioned large files.
	- External storage (S3, GDrive, institutional storage) and store URIs in your experiment metadata.
	- Keeping package builds (like a full PyTorch wheel) out of git; install them inside the `.uv` venv as needed.


