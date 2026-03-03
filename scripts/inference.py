"""Inference pipeline for lightning nowcasting with pretrained models.

This script reproduces the paper results by:
  1. (Optionally) downloading dataset and pretrained weights from Zenodo
  2. Setting up the data batch generator
  3. Loading the pretrained ensemble model
  4. Running inference on the test (or validation) set
  5. Saving confusion-matrix results to the results/ directory

Dataset and pretrained weights:
  https://doi.org/10.5281/zenodo.6325370

Usage examples
--------------
# Download everything automatically, then evaluate on the test set:
  python inference.py --download

# Use already-downloaded files in custom directories:
  python inference.py --data-dir /path/to/data --model-dir /path/to/models

# Evaluate on the validation set instead:
  python inference.py --dataset valid

# Compute lead-time metrics as well:
  python inference.py --leadtime-metrics
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

# Allow running from inside scripts/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training import setup_batch_gen, build_ensemble_model
from c4dllightning.analysis.evaluation import (
    confusion_matrix,
    conf_matrix_leadtimes,
)


# ---------------------------------------------------------------------------
# Zenodo helpers
# ---------------------------------------------------------------------------

ZENODO_RECORD_ID = "6325370"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Weight files required by build_ensemble_model (dropout variant)
DROPOUT_WEIGHT_FILES = [
    "lightning_dropout_weightdecay_noclassweight.h5",
    "lightning_dropout_weightdecay_noclassweight2.h5",
    "lightning_dropout_weightdecay_noclassweight3.h5",
]
# Weight files for the non-dropout ensemble variant
NODROPOUT_WEIGHT_FILES = [
    "lightning_noclassweight1.h5",
    "lightning_noclassweight2.h5",
    "lightning_noclassweight3.h5",
]


def _fetch_zenodo_files():
    """Return the list of file records from the Zenodo API."""
    try:
        with urllib.request.urlopen(ZENODO_API_URL, timeout=30) as resp:
            record = json.loads(resp.read())
        return record.get("files", [])
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the Zenodo API ({ZENODO_API_URL}): {exc}\n"
            "Please download the files manually from "
            "https://doi.org/10.5281/zenodo.6325370"
        ) from exc


def _download_file(url, dest):
    """Download *url* to *dest*, showing a simple progress indicator."""
    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r    {pct}% ", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress


def download_zenodo(dest_data_dir, dest_model_dir, file_suffix="2020"):
    """Download all dataset files and model weights from Zenodo.

    Parameters
    ----------
    dest_data_dir : str
        Directory where NetCDF patch files will be saved.
    dest_model_dir : str
        Directory where model weight (.h5) files will be saved.
    file_suffix : str
        Year suffix used in data filenames (e.g. ``'2020'``).
    """
    os.makedirs(dest_data_dir, exist_ok=True)
    os.makedirs(dest_model_dir, exist_ok=True)

    print("Fetching file list from Zenodo …")
    files = _fetch_zenodo_files()

    if not files:
        raise RuntimeError(
            "No files found in the Zenodo record. "
            "Please check https://doi.org/10.5281/zenodo.6325370 manually."
        )

    for f in files:
        key = f["key"]
        url = f["links"]["self"]

        # Determine destination directory based on file type
        if key.endswith(".h5"):
            dest = os.path.join(dest_model_dir, key)
        elif key.endswith(".nc") and (
            key.startswith("patches") or file_suffix in key
        ):
            dest = os.path.join(dest_data_dir, key)
        else:
            # Skip files we don't recognise (e.g. documentation PDFs)
            continue

        if os.path.exists(dest):
            print(f"  {key} already present, skipping.")
        else:
            _download_file(url, dest)

    print("Download complete.")


# ---------------------------------------------------------------------------
# Inference / evaluation
# ---------------------------------------------------------------------------


def run_inference(
    data_dir,
    model_dir,
    results_dir,
    dataset="test",
    dropout=True,
    batch_size=48,
    file_suffix="2020",
    leadtime_metrics=False,
):
    """Run inference with the pretrained ensemble model.

    Parameters
    ----------
    data_dir : str
        Directory containing the NetCDF patch files.
    model_dir : str
        Directory containing the pretrained ``.h5`` weight files.
    results_dir : str
        Directory where NumPy result arrays are written.
    dataset : str
        Which split to evaluate: ``'test'`` or ``'valid'``.
    dropout : bool
        ``True`` → use the dropout-regularised ensemble (recommended);
        ``False`` → use the non-dropout ensemble.
    batch_size : int
        Inference batch size.
    file_suffix : str
        Year suffix used in data filenames (e.g. ``'2020'``).
    leadtime_metrics : bool
        If ``True``, also compute and save per-lead-time confusion matrices.
    """
    # ------------------------------------------------------------------
    # Validate weight files
    # ------------------------------------------------------------------
    weight_files = DROPOUT_WEIGHT_FILES if dropout else NODROPOUT_WEIGHT_FILES
    for wf in weight_files:
        path = os.path.join(model_dir, wf)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model weight file not found: {path}\n"
                "Run with --download to fetch the pretrained weights, or "
                "place the .h5 files in the models/ directory manually."
            )

    # ------------------------------------------------------------------
    # Step 1 – Build data batch generator
    # ------------------------------------------------------------------
    print("Step 1/3 – Setting up batch generator …")
    batch_gen = setup_batch_gen(
        data_dir,
        file_suffix=file_suffix,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Step 2 – Load pretrained ensemble model
    # ------------------------------------------------------------------
    print("Step 2/3 – Loading pretrained ensemble model …")
    (model, _strategy) = build_ensemble_model(
        batch_gen, dropout=dropout, model_dir=model_dir
    )

    # ------------------------------------------------------------------
    # Step 3 – Evaluate on requested split
    # ------------------------------------------------------------------
    # Upper bound slightly > 1.0 to include 1.0 in the range despite floating-point rounding
    thresholds = np.arange(0, 1.0001, 0.001)
    out_dir = os.path.join(results_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    tag = (
        "ensemble_dropout_weightdecay_noclassweight"
        if dropout
        else "ensemble_noclassweight"
    )

    print(f"Step 3/3 – Running inference on '{dataset}' set …")
    cm = confusion_matrix(
        model, batch_gen, dataset=dataset, thresholds=thresholds
    )
    cm_path = os.path.join(out_dir, f"conf_matrix-{tag}.npy")
    np.save(cm_path, cm)
    print(f"  Saved confusion matrix → {cm_path}")

    if leadtime_metrics:
        print("  Computing per-lead-time metrics …")
        cm_lt = conf_matrix_leadtimes(
            model,
            batch_gen,
            dataset=dataset,
            thresholds=thresholds,
            num_leadtimes=batch_gen.timesteps[1],
        )
        cm_lt_path = os.path.join(
            out_dir, f"conf_matrix_leadtime-{tag}.npy"
        )
        np.save(cm_lt_path, cm_lt)
        print(f"  Saved lead-time confusion matrix → {cm_lt_path}")

    # ------------------------------------------------------------------
    # Print summary metrics at threshold = 0.5
    # ------------------------------------------------------------------
    thresh_idx = int(np.argmin(np.abs(thresholds - 0.5)))
    ((tp, fn), (fp, tn)) = cm[:, :, thresh_idx]
    denom_prec = tp + fp
    denom_rec = tp + fn
    denom_csi = tp + fp + fn
    precision = (tp / denom_prec) if denom_prec > 0 else 0.0
    recall = (tp / denom_rec) if denom_rec > 0 else 0.0
    csi = (tp / denom_csi) if denom_csi > 0 else 0.0

    print(f"\nSummary metrics at threshold = 0.5 ('{dataset}' set):")
    print(f"  Precision (1-FAR) : {precision:.4f}")
    print(f"  Recall (POD)      : {recall:.4f}")
    print(f"  CSI               : {csi:.4f}")

    return cm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce lightning nowcasting results using pretrained models. "
            "Data and weights are available at "
            "https://doi.org/10.5281/zenodo.6325370"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_REPO_ROOT, "data"),
        help="Directory containing (or to store) the NetCDF patch files.",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(_REPO_ROOT, "models"),
        help="Directory containing (or to store) the pretrained .h5 weight files.",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_REPO_ROOT, "results"),
        help="Directory where evaluation results (NumPy arrays) are written.",
    )
    parser.add_argument(
        "--dataset",
        default="test",
        choices=["test", "valid"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        help="Use the non-dropout ensemble instead of the dropout ensemble.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download dataset and pretrained weights from Zenodo "
            "(https://doi.org/10.5281/zenodo.6325370) before running inference."
        ),
    )
    parser.add_argument(
        "--file-suffix",
        default="2020",
        help="Year suffix appended to data filenames (e.g. '2020').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--leadtime-metrics",
        action="store_true",
        help="Also compute and save per-lead-time confusion matrices.",
    )

    args = parser.parse_args()

    if args.download:
        download_zenodo(
            dest_data_dir=args.data_dir,
            dest_model_dir=args.model_dir,
            file_suffix=args.file_suffix,
        )

    run_inference(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        dataset=args.dataset,
        dropout=not args.no_dropout,
        batch_size=args.batch_size,
        file_suffix=args.file_suffix,
        leadtime_metrics=args.leadtime_metrics,
    )


if __name__ == "__main__":
    main()
