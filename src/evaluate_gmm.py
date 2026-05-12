import argparse
import csv
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from experiment_utils import (
    create_run_dir,
    experiment_params,
    load_config,
    resolve_run_dir,
    save_json,
    validate_config,
)
from losses import reconstruction_error_per_sample
from model import Autoencoder
from tracker_integration import log_run


def extract_latent_vectors(model, loader, device, latent_dim):
    latents = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
    if not latents:
        return np.zeros((0, latent_dim), dtype=np.float32)
    return np.concatenate(latents, axis=0)


def extract_latents_and_recon_errors(
    model,
    loader,
    device,
    latent_dim,
    reconstruction_loss_mode,
    reconstruction_target,
):
    latents = []
    recon_errors = []
    labels = []
    with torch.no_grad():
        for x, batch_labels in loader:
            x = x.to(device)
            z = model.encode(x)
            x_recon = model.decoder(z)
            batch_errors = reconstruction_error_per_sample(
                x_recon,
                x,
                mode=reconstruction_loss_mode,
                target=reconstruction_target,
            )
            latents.append(z.cpu().numpy())
            recon_errors.extend(batch_errors.cpu().numpy())
            labels.extend(batch_labels.numpy())

    if not latents:
        return (
            np.zeros((0, latent_dim), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64),
        )
    return (
        np.concatenate(latents, axis=0),
        np.array(recon_errors, dtype=np.float64),
        np.array(labels, dtype=np.int64),
    )


def normalize_with_reference(values, reference_values):
    ref_min = float(np.min(reference_values))
    ref_max = float(np.max(reference_values))
    denom = ref_max - ref_min
    if denom <= 1e-12:
        return np.zeros_like(values, dtype=np.float64), ref_min, ref_max
    return (values - ref_min) / denom, ref_min, ref_max


def combine_scores(gmm_scores, recon_errors, normal_gmm_scores, normal_recon_errors, alpha):
    normalized_gmm, gmm_min, gmm_max = normalize_with_reference(gmm_scores, normal_gmm_scores)
    normalized_recon, recon_min, recon_max = normalize_with_reference(recon_errors, normal_recon_errors)
    final_scores = alpha * normalized_gmm + (1 - alpha) * normalized_recon
    normalization = {
        'gmm_score_min': gmm_min,
        'gmm_score_max': gmm_max,
        'reconstruction_error_min': recon_min,
        'reconstruction_error_max': recon_max,
    }
    return final_scores, normalized_gmm, normalized_recon, normalization


def make_dataset(config, split, include_pneumonia=True):
    return ChestXrayDataset(
        config['data_dir'],
        split,
        image_size=config['image_size'],
        spectral_mode=config['spectral_mode'],
        high_freq_cutoff_ratio=config['high_freq_cutoff_ratio'],
        include_pneumonia=include_pneumonia,
    )


def select_elbow_k(results, score_key='BIC'):
    if len(results) < 3:
        return int(min(results, key=lambda row: row[score_key])['K'])

    ks = np.array([row['K'] for row in results], dtype=np.float64)
    scores = np.array([row[score_key] for row in results], dtype=np.float64)

    k_range = ks.max() - ks.min()
    score_range = scores.max() - scores.min()
    if k_range <= 0 or score_range <= 1e-12:
        return int(min(results, key=lambda row: row[score_key])['K'])

    points = np.column_stack(((ks - ks.min()) / k_range, (scores - scores.min()) / score_range))
    start = points[0]
    end = points[-1]
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm <= 1e-12:
        return int(min(results, key=lambda row: row[score_key])['K'])

    # Elbow method: choose the point with the largest perpendicular distance
    # from the line connecting the first and last BIC points. This captures the
    # diminishing-return point before extra mixture components add complexity.
    deltas = points - start
    distances = np.abs(line[0] * deltas[:, 1] - line[1] * deltas[:, 0]) / line_norm
    return int(ks[int(np.argmax(distances))])


def save_gmm_selection(results, csv_path, plot_path, best_k=None, elbow_k=None):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['K', 'BIC', 'AIC'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    ks = [row['K'] for row in results]
    bics = [row['BIC'] for row in results]
    aics = [row['AIC'] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, bics, marker='o', label='BIC')
    plt.plot(ks, aics, marker='o', label='AIC')
    if elbow_k is not None:
        elbow_bic = next(row['BIC'] for row in results if row['K'] == elbow_k)
        plt.scatter([elbow_k], [elbow_bic], s=90, color='red', zorder=5, label=f'Elbow K={elbow_k}')
    if best_k is not None and best_k != elbow_k:
        best_bic = next(row['BIC'] for row in results if row['K'] == best_k)
        plt.scatter([best_k], [best_bic], s=80, color='black', zorder=5, label=f'Selected K={best_k}')
    plt.xticks(ks)
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('GMM Model Selection: BIC and AIC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def save_score_histogram(normal_scores, pneumonia_scores, threshold, plot_path):
    plt.figure(figsize=(8, 6))
    plt.hist(normal_scores, alpha=0.55, label='NORMAL', bins=50)
    plt.hist(pneumonia_scores, alpha=0.55, label='PNEUMONIA', bins=50)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Final Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Final Score Distribution')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection using GMM on AE latent vectors')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML experiment config')
    parser.add_argument('--data_dir', type=str, help='Override data/chest_xray path from config')
    return parser.parse_args()


def main():
    args = parse_args()
    config = validate_config(load_config(args.config))
    config.setdefault('data_dir', 'data/chest_xray')
    if args.data_dir:
        config['data_dir'] = args.data_dir

    data_dir = config['data_dir']
    required_dirs = [
        os.path.join(data_dir, 'train', 'NORMAL'),
        os.path.join(data_dir, 'val', 'NORMAL'),
        os.path.join(data_dir, 'test', 'NORMAL'),
        os.path.join(data_dir, 'test', 'PNEUMONIA'),
    ]
    missing = [path for path in required_dirs if not os.path.exists(path)]
    if missing:
        print(f'Error: expected dataset structure under {data_dir}:')
        print('  train/NORMAL/')
        print('  val/NORMAL/')
        print('  test/NORMAL/')
        print('  test/PNEUMONIA/')
        print(f'Missing: {missing}')
        return

    run_dir = resolve_run_dir(args.config)
    if run_dir is None:
        run_id, run_dir = create_run_dir(config)
        print(f'No prior training run found for {args.config}; created evaluation run {run_id}')
    else:
        run_id = run_dir.name

    checkpoint_dir = run_dir / 'checkpoints'
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'ae.pt'
    if not checkpoint_path.exists():
        print(f'Error: missing AE checkpoint for this config: {checkpoint_path}')
        print('Run training first, for example:')
        print(f'  python src/train_ae.py --config {args.config}')
        return

    params = experiment_params(config, run_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=int(config['latent_dim']), image_size=int(config['image_size']))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    train_dataset = make_dataset(config, 'train', include_pneumonia=False)
    val_dataset = make_dataset(config, 'val')
    test_dataset = make_dataset(config, 'test')

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print('Error: train, val, and test datasets must all contain samples.')
        return

    batch_size = int(config['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # The GMM is fit on AE latent vectors rather than pixels. The encoder
    # compresses the spectrum into learned normal-pattern features, making the
    # density model lower-dimensional and less sensitive to pixel-level noise.
    z_train = extract_latent_vectors(model, train_loader, device, int(config['latent_dim']))
    if z_train.shape[0] == 0:
        print('Error: failed to extract latent vectors from training data')
        return

    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train).astype(np.float64)

    model_selection = []
    min_k, max_k = [int(value) for value in config['gmm_k_range']]
    for k in range(min_k, max_k + 1):
        if k > z_train_scaled.shape[0]:
            print(f'Skipping K={k} because it exceeds number of training samples ({z_train_scaled.shape[0]})')
            break

        candidate = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            reg_covar=1e-5,
            n_init=3,
            tol=1e-3,
        )
        try:
            candidate.fit(z_train_scaled)
        except ValueError as exc:
            print(f'Warning: skipped K={k} due to GMM fit failure: {exc}')
            continue

        model_selection.append({
            'K': k,
            'BIC': float(candidate.bic(z_train_scaled)),
            'AIC': float(candidate.aic(z_train_scaled)),
        })
        print(f'K={k}: BIC={model_selection[-1]["BIC"]:.1f}, AIC={model_selection[-1]["AIC"]:.1f}')

    if len(model_selection) == 0:
        print('Error: no valid GMM models could be fit to the training latent vectors.')
        return

    min_bic_k = int(min(model_selection, key=lambda row: row['BIC'])['K'])
    elbow_k = select_elbow_k(model_selection, score_key='BIC')
    selection_method = config.get('gmm_selection_method', 'bic')
    # BIC balances likelihood against model complexity. The optional elbow mode
    # still uses the BIC curve, but chooses the diminishing-return point instead
    # of the absolute minimum.
    best_k = elbow_k if selection_method == 'elbow' else min_bic_k
    params['best_K'] = best_k
    params['min_bic_K'] = min_bic_k
    params['elbow_K'] = elbow_k
    print(f'Minimum-BIC K: {min_bic_k}')
    print(f'Elbow K: {elbow_k}')
    print(f'Selected best K by {selection_method}: {best_k}')

    save_gmm_selection(
        model_selection,
        run_dir / 'gmm_model_selection.csv',
        run_dir / 'gmm_bic_aic.png',
        best_k=best_k,
        elbow_k=elbow_k,
    )

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type='full',
        random_state=42,
        reg_covar=1e-5,
        n_init=3,
        tol=1e-3,
    )
    gmm.fit(z_train_scaled)
    score_alpha = float(config.get('score_alpha', 1.0))
    reconstruction_loss_mode = config.get('reconstruction_loss', 'mse')
    reconstruction_target = config.get('reconstruction_target', 'all_channels')
    joblib.dump(
        {
            'scaler': scaler,
            'gmm': gmm,
            'best_k': best_k,
            'score_alpha': score_alpha,
            'reconstruction_loss': reconstruction_loss_mode,
            'reconstruction_target': reconstruction_target,
        },
        checkpoint_dir / 'gmm_latent.joblib',
    )

    z_val, val_recon_errors, val_labels = extract_latents_and_recon_errors(
        model,
        val_loader,
        device,
        int(config['latent_dim']),
        reconstruction_loss_mode,
        reconstruction_target,
    )
    normal_val_mask = val_labels == 0
    if not np.any(normal_val_mask):
        print('Error: no NORMAL samples in val/NORMAL for threshold selection')
        return

    z_val_scaled = scaler.transform(z_val).astype(np.float64)
    val_gmm_scores = -gmm.score_samples(z_val_scaled)
    val_final_scores, val_gmm_norm, val_recon_norm, score_normalization = combine_scores(
        val_gmm_scores,
        val_recon_errors,
        val_gmm_scores[normal_val_mask],
        val_recon_errors[normal_val_mask],
        score_alpha,
    )
    # The threshold is learned from the upper tail of validation NORMAL scores.
    # That fixes the false-positive operating point without looking at test
    # pneumonia examples.
    threshold = float(np.percentile(val_final_scores[normal_val_mask], float(config['threshold_percentile'])))
    params['threshold'] = threshold
    params['score_normalization'] = score_normalization
    print(
        f'Threshold set from val/NORMAL {config["threshold_percentile"]}th percentile '
        f'on final score: {threshold:.4f}'
    )
    print(
        f'Final score = {score_alpha:.2f} * normalized GMM score + '
        f'{1 - score_alpha:.2f} * normalized {reconstruction_loss_mode} '
        f'({reconstruction_target})'
    )

    z_test, test_recon_errors, labels = extract_latents_and_recon_errors(
        model,
        test_loader,
        device,
        int(config['latent_dim']),
        reconstruction_loss_mode,
        reconstruction_target,
    )
    z_test_scaled = scaler.transform(z_test).astype(np.float64)
    gmm_scores = -gmm.score_samples(z_test_scaled)
    final_scores, gmm_scores_norm, recon_errors_norm, _ = combine_scores(
        gmm_scores,
        test_recon_errors,
        val_gmm_scores[normal_val_mask],
        val_recon_errors[normal_val_mask],
        score_alpha,
    )
    predictions = (final_scores > threshold).astype(int)

    roc_auc = roc_auc_score(labels, final_scores)
    precision_vals, recall_vals, _ = precision_recall_curve(labels, final_scores)
    pr_auc = auc(recall_vals, precision_vals)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    ) * 100.0
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    normal_scores = final_scores[labels == 0]
    pneumonia_scores = final_scores[labels == 1]
    hist_path = plots_dir / 'gmm_score_hist_threshold.png'
    save_score_histogram(normal_scores, pneumonia_scores, threshold, hist_path)

    metrics = {
        'best_K': int(best_k),
        'min_bic_K': int(min_bic_k),
        'elbow_K': int(elbow_k),
        'gmm_selection_method': selection_method,
        'score_alpha': float(score_alpha),
        'reconstruction_loss': reconstruction_loss_mode,
        'reconstruction_target': reconstruction_target,
        'threshold': float(threshold),
        'mean_gmm_score': float(np.mean(gmm_scores)),
        'mean_reconstruction_error': float(np.mean(test_recon_errors)),
        'mean_final_score': float(np.mean(final_scores)),
        'score_normalization': score_normalization,
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_percent': np.round(cm_percent, 2).tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    save_json(run_dir / 'params.json', params)
    save_json(run_dir / 'metrics.json', metrics)

    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'PR-AUC: {pr_auc:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    print('Confusion Matrix (% by actual class):')
    print(np.array2string(cm_percent, formatter={'float_kind': lambda value: f'{value:6.2f}'}))
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    log_run(
        experiment_name=config['experiment_name'],
        parameters=params,
        metrics=metrics,
        notes=(
            'AE latent vectors are standardized, GMM K is selected by minimum BIC, '
            'and the anomaly threshold is the configured validation-normal percentile.'
        ),
        tags=['evaluation', 'anomaly_detection', 'gmm', config['spectral_mode'], 'chest_xray'],
        artifacts=[
            str(run_dir / 'config_snapshot.yaml'),
            str(run_dir / 'params.json'),
            str(run_dir / 'metrics.json'),
            str(run_dir / 'gmm_model_selection.csv'),
            str(run_dir / 'gmm_bic_aic.png'),
            str(hist_path),
            str(checkpoint_dir / 'gmm_latent.joblib'),
        ],
    )

    print(f'Run ID: {run_id}')
    print(f'Run outputs: {run_dir}')


if __name__ == '__main__':
    main()
