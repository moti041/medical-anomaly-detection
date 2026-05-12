# Experiment Tracking Guide

The chest X-ray anomaly detection project is integrated with **ml_flow_like**, a lightweight experiment tracker for comparing and analyzing model runs.

## Quick Start

### 1. **Train and Log**
```bash
python src/train_ae.py --config configs/ae_full_spectrum.yaml
python src/train_ae.py --config configs/ae_high_freq.yaml
```
This automatically logs:
- Training parameters (epochs, batch size, learning rate, device)
- Final loss value
- Model checkpoint files
- `outputs/{run_id}/config_snapshot.yaml`, `params.json`, `metrics.json`, and `checkpoints/`

### 2. **Evaluate and Compare**
```bash
python src/evaluate_gmm.py --config configs/ae_full_spectrum.yaml
python src/evaluate_gmm.py --config configs/ae_high_freq.yaml
```
This automatically logs:
- Evaluation metrics (ROC-AUC, PR-AUC, precision, recall, F1)
- Threshold value and confusion matrix
- Score distribution plot
- Confusion matrix breakdown (TP, TN, FP, FN)
- GMM BIC/AIC model-selection plot and CSV

### 3. **View Results in Web UI**
```bash
python src/experiment_cli.py ui
```
Opens the tracker dashboard at `http://127.0.0.1:8000`

Features:
- **Compare runs side-by-side**: Select multiple runs to compare parameters and metrics
- **Plot metrics**: Visualize metrics (ROC-AUC, loss, etc.) across runs
- **View artifacts**: Download saved plots, model files, etc.
- **Generate reports**: Create PowerPoint comparisons between two runs

## Workflow Examples

### Example 1: Try Different Thresholds
```bash
# Edit threshold_percentile in a copied config, then run:
python src/train_ae.py --config configs/ae_full_spectrum.yaml
python src/evaluate_gmm.py --config configs/ae_full_spectrum.yaml
```
Then in the UI, compare all 3 runs to find the best threshold.

### Example 2: Compare Model Architectures
```bash
# Baseline AE
python src/train_ae.py --config configs/ae_full_spectrum.yaml

# High-frequency spectrum AE
python src/train_ae.py --config configs/ae_high_freq.yaml
```
Compare final loss values to see which architecture trains better.

### Example 3: Tune Hyperparameters
Copy a config, change one hyperparameter, then run:
```bash
python src/train_ae.py --config configs/my_trial.yaml
python src/evaluate_gmm.py --config configs/my_trial.yaml
```
Use the UI to plot loss across epochs for each run.

## CLI Commands

### List All Runs
```bash
python src/experiment_cli.py list
```
Shows: Run ID, Experiment Name, Timestamp, Tags

### Open Tracker UI
```bash
python src/experiment_cli.py ui
```
Starts backend and opens browser to `http://127.0.0.1:8000`

## What Gets Logged Automatically

### Training Runs
- **Experiment Name**: from `experiment_name` in the YAML config
- **Parameters**: run_id, config filename, spectral_mode, cutoff_ratio, image_size, batch_size, epochs, learning_rate, latent_dim, git commit hash if available
- **Metrics**: final_loss, total_samples
- **Artifacts**: ae.pt, ae_best.pt
- **Tags**: training, autoencoder, chest_xray

### Evaluation Runs
- **Experiment Name**: from `experiment_name` in the YAML config
- **Parameters**: run_id, config filename, spectral_mode, cutoff_ratio, latent_dim, best_K, threshold, git commit hash if available
- **Metrics**: roc_auc, pr_auc, precision, recall, f1, specificity, threshold, confusion_matrix, TP/TN/FP/FN
- **Artifacts**: gmm_bic_aic.png, gmm_model_selection.csv, plots/gmm_score_hist_threshold.png, gmm_latent.joblib
- **Tags**: evaluation, anomaly_detection, gmm, chest_xray

## Performance Benchmarking Workflow

1. **Train full spectrum**: `python src/train_ae.py --config configs/ae_full_spectrum.yaml`
2. **Evaluate full spectrum**: `python src/evaluate_gmm.py --config configs/ae_full_spectrum.yaml`
3. **Train high frequency**: `python src/train_ae.py --config configs/ae_high_freq.yaml`
4. **Evaluate high frequency**: `python src/evaluate_gmm.py --config configs/ae_high_freq.yaml`
5. **Compare in UI**: Open tracker UI and select both runs to see side-by-side metrics
6. **Analyze**: Check which configuration gives best ROC-AUC / Recall trade-off
7. **Export report**: Generate PowerPoint comparison for documentation

Keep all fields identical between configs except `experiment_name` and the intended spectral settings. That makes the full-spectrum and high-frequency runs comparable.

## Method Notes

- The GMM is fit on autoencoder latent vectors because the encoder compresses FFT images into learned normal-pattern features. This gives the density model a lower-dimensional, less noisy space than raw pixels.
- K selection is controlled by `gmm_selection_method`. `bic` chooses the minimum BIC. `elbow` uses the BIC curve and chooses the point with the largest distance from the line between the first and last K values, which captures the diminishing-return point.
- The final anomaly score is controlled by `score_alpha`: `score_alpha * normalized_gmm_score + (1 - score_alpha) * normalized_reconstruction_error`. The default configs currently use `0.5`, which gives equal weight to both signals.
- The reconstruction term is controlled by `reconstruction_loss` and `reconstruction_target`. `energy_normalized_mse` uses `mean((x_recon - x)^2) / (mean(x^2) + eps)`. With `reconstruction_target: fft_only`, the AE receives image plus FFT channels but is trained and scored only on the FFT channel.
- The anomaly threshold comes from the upper tail of `val/NORMAL` scores so the operating point is selected without looking at pneumonia test examples.
- High-frequency FFT experiments may help when anomalies change local texture, edge sharpness, or fine-grained lung opacity patterns more than global image structure.

## Advanced Usage

### Access Tracker Programmatically
```python
from tracker_integration import log_run

log_run(
    experiment_name="my_experiment",
    parameters={"param1": 0.001, "param2": 32},
    metrics={"metric1": 0.95, "metric2": 0.88},
    notes="Testing configuration XYZ",
    tags=["test", "v2"],
    artifacts=["path/to/plot.png", "path/to/model.pt"]
)
```

### Direct Database Access
```python
import sys
sys.path.insert(0, "C:\\Users\\Public\\ml_flow_like")
from backend.repository import list_runs, get_run

all_runs = list_runs()
specific_run = get_run("abc123def456")
print(specific_run["metrics"])
```

## Tips for Effective Benchmarking

1. **Use meaningful tags**: Label runs with version/config (e.g., "v1", "with_dropout", "lr_0.0001")
2. **Write notes**: Document what changed from previous run (e.g., "added batch norm to decoder")
3. **Save artifacts**: Always save key files (plots, models) for later analysis
4. **Compare incrementally**: Make one change at a time to isolate effects
5. **Track metrics over time**: Plot key metrics (loss, ROC-AUC) to see progress
6. **Generate reports**: Use the tracker to create comparison reports before finalizing experiments

## Troubleshooting

### Tracker UI not opening
- Make sure port 8000 is available: `netstat -ano | findstr :8000`
- Kill any existing process: `taskkill /PID <pid> /F`

### Artifacts not saving
- Check that output directories exist: `outputs/`, `checkpoints/`
- Ensure file paths are absolute or relative to project root

### No runs showing up
- Run `python src/experiment_cli.py list` to check database
- Check that tracker_integration.py is imported correctly

## See Also
- [ml_flow_like Documentation](C:\Users\Public\ml_flow_like\README.md)
- [example_log_run.py](C:\Users\Public\ml_flow_like\example_log_run.py)
