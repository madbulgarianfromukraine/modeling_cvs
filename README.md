# CNN Thesis Project

Repository for training and fine-tuning convolutional neural networks.

> **Note:** This repository is part of a broader scientific work. You can read the full bachelor thesis and research findings here: [Bachelor Thesis: Modeling CVS](https://madbulgarianfromukraine.github.io/bachelor-thesis-modeling-cvs/)

## Structure
- `src/` - Library code, dataset utilities, and analysis notebooks. See `src/README.md` for details on structure and notebooks.
- `demo/` - Standalone demo scripts for visualizations. See `demo/README.md` for execution instructions.
- `tests/` - Unit tests and smoke tests.
- `data/` - Raw and processed datasets (gitignored by default).

## Setup

To set up the project locally:

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Pipelines

The primary workflows are driven through the Jupyter notebooks located in `src/`. 

- **Model Pipelines**: Run `model.ipynb` (if available) or `src/simple_model/analysis.ipynb` for main analysis.
- **Data Analysis**: Run `src/data/dataset_analysis.ipynb`.

**Note on Kaggle environments:** If you are running these pipelines in Kaggle notebooks, ensure you mount the correct input paths and adjust dataset directories accordingly, since Kaggle uses a different file structure (e.g., `/kaggle/input/`).

## Tests

To run the test suite, use pytest from the root of the repository:
```bash
python -m pytest
```
