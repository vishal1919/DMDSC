# DMDSC: A Dynamic-Margin Deep Simplex Classifier for Open-Set Recognition on Medical Image Datasets

This repository contains the implementation code for the paper "DMDSC: A Dynamic-Margin Deep Simplex Classifier for Open-Set Recognition on Medical Image Datasets".

## Dataset Setup

### MedMNIST Dataset
The MedMNIST dataset can be downloaded from:
- **Official source**: https://zenodo.org/records/10519652

### Background Data
For extended experiments, download the 300K random images:
- **Source**: https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy

### Basic Training
Run the main training script with default parameters:
```bash
python NirvanaOSR.py --dataset dataset-name --dataroot ./data --outf ./results
```

### Hyperparameter Configuration

| Parameter | Description |
|-----------|-------------|
| `--batch-size` | Training batch size | 
| `--lr` | Learning rate | 
| `--max-epoch` | Maximum training epochs |
| `--optim` | Optimizer to be used |
| `--m-min` | Minimum margin for loss | 
| `--m-max` | Maximum margin for loss | 
| `--Expand` | Expand factor of centers |
| `--inter-weight` | Weight for inter-class loss |
| `--outlier-weight` | Weight for outlier triplet loss |
| `--model` | Backbone network to be used |



## Acknowledgments

This codebase is derived from the Deep Simplex Classifier implementation available at https://github.com/Cevikalp/dsc. We thank the authors for making their code available.


