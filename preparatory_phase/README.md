# Preparatory phase notes

Use this folder for: exploratory scripts, quick experiments, dataset inspection, and prototypes before integrating into `main/`.

## Experiments & training notes

We first had a vanilla PseudAlexNet, but it gave poorer results.

After that we increased the momentum of the SGD minimizer and used batchnorms after almost all layers. Then we fine-tuned the documents dataset on it. Fine-tuning happened in two stages:

1. Warm-up stage — retrain the fully connected layers for 20 epochs or until convergence.
2. Drift stage — allow retraining of all layers except the first 2 convolutional layers.

All results and console output are stored in the corresponding subfolders under the `results/` folder. Training runs were executed in Kaggle Notebooks using Google's TPU v5-e8.
Preparatory phase notes

Use this folder for: exploratory scripts, quick experiments, dataset inspection, and prototypes before integrating into `main/`.
