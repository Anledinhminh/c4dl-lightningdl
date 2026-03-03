This repository contains the machine learning code used in the paper: Multi-source seamless lightning nowcasting with recurrent deep learning, accepted to _Artificial Intelligence for the Earth Systems_. Preprint available: https://arxiv.org/abs/2203.10114

# Installation

You need NumPy, Scipy, Matplotlib, Tensorflow, Numba, Dask and NetCDF4 for Python.

Clone the repository, then, in the main directory, run
```bash
$ python setup.py develop
```
(if you plan to modify the code) or
```bash
$ python setup.py install
```
if you just want to run it.

# Downloading data

The dataset and pretrained model weights can be found at the following Zenodo repository: https://doi.org/10.5281/zenodo.6325370

Download the NetCDF patch files and the `.h5` model weight files. You can place the NetCDF files in the `data` directory and the `.h5` files in the `models` directory (or use the `--download` flag described below).

# Reproducing results with pretrained models

The `scripts/inference.py` script provides a complete pipeline to reproduce the paper results using the pretrained ensemble model:

**Quick start – let the script download everything automatically:**
```bash
cd scripts
python inference.py --download
```

This will:
1. Download the dataset and pretrained weights from Zenodo into `data/` and `models/`.
2. Set up the data batch generator.
3. Load the pretrained dropout ensemble model.
4. Run inference on the **test** set and print Precision, Recall, and CSI at threshold 0.5.
5. Save the confusion matrix to `results/test/conf_matrix-ensemble_dropout_weightdecay_noclassweight.npy`.

**Using manually downloaded files:**
```bash
cd scripts
python inference.py --data-dir /path/to/data --model-dir /path/to/models
```

**Additional options:**
```
  --dataset {test,valid}   Evaluate on the test or validation split (default: test)
  --leadtime-metrics       Also compute and save per-lead-time confusion matrices
  --no-dropout             Use the non-dropout ensemble instead
  --batch-size N           Inference batch size (default: 48)
  --file-suffix YEAR       Year suffix in data filenames (default: 2020)
```

After evaluation you can reproduce all plots from the paper by running `plots_lightning.py` in an interactive shell from the `scripts` directory.

# Training from scratch

Go to the `scripts` directory and start an interactive shell. There, you can find `training.py` that contains the script you need for training and `plots_lightning.py` that produces the plots from the paper.
