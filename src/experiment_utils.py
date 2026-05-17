import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    config['config_filename'] = config_path.name
    config['config_path'] = str(config_path)
    return config


def validate_config(config):
    required = [
        'experiment_name',
        'spectral_mode',
        'image_size',
        'batch_size',
        'epochs',
        'learning_rate',
        'latent_dim',
        'gmm_k_range',
        'threshold_percentile',
        'high_freq_cutoff_ratio',
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if config['spectral_mode'] not in {'full_spectrum', 'high_freq'}:
        raise ValueError("spectral_mode must be 'full_spectrum' or 'high_freq'")
    if not isinstance(config['gmm_k_range'], list) or len(config['gmm_k_range']) != 2:
        raise ValueError('gmm_k_range must be a two-item list, for example [1, 6]')
    if int(config['gmm_k_range'][0]) < 1 or int(config['gmm_k_range'][1]) < int(config['gmm_k_range'][0]):
        raise ValueError('gmm_k_range must be an increasing positive range')
    config.setdefault('gmm_selection_method', 'elbow')
    if config['gmm_selection_method'] not in {'bic', 'elbow'}:
        raise ValueError("gmm_selection_method must be 'bic' or 'elbow'")
    config.setdefault('score_alpha', 1.0)
    alpha = float(config['score_alpha'])
    if not 0 <= alpha <= 1:
        raise ValueError('score_alpha must be between 0 and 1')
    config.setdefault('reconstruction_loss', 'mse')
    if config['reconstruction_loss'] not in {'mse', 'energy_normalized_mse'}:
        raise ValueError("reconstruction_loss must be 'mse' or 'energy_normalized_mse'")
    config.setdefault('reconstruction_target', 'all_channels')
    if config['reconstruction_target'] not in {'all_channels', 'fft_only'}:
        raise ValueError("reconstruction_target must be 'all_channels' or 'fft_only'")
    config.setdefault('use_pca', False)
    if not isinstance(config['use_pca'], bool):
        raise ValueError('use_pca must be true or false')
    config.setdefault('pca_n_components', 0.95)
    pca_n_components = config['pca_n_components']
    if isinstance(pca_n_components, float):
        if not 0 < pca_n_components <= 1:
            raise ValueError('pca_n_components float must be in (0, 1]')
    elif isinstance(pca_n_components, int):
        if pca_n_components < 1:
            raise ValueError('pca_n_components int must be >= 1')
    else:
        raise ValueError('pca_n_components must be float or int')
    config.setdefault('test_subset', 'all_pneumonia')
    if config['test_subset'] not in {'all_pneumonia', 'virus_only', 'bacteria_only'}:
        raise ValueError("test_subset must be 'all_pneumonia', 'virus_only', or 'bacteria_only'")
    config.setdefault('activation', 'relu')
    config['activation'] = str(config['activation']).lower()
    if config['activation'] not in {'relu', 'leaky_relu'}:
        raise ValueError("activation must be 'relu' or 'leaky_relu'")
    config.setdefault('leaky_relu_slope', 0.1)
    if float(config['leaky_relu_slope']) < 0:
        raise ValueError('leaky_relu_slope must be non-negative')
    cutoff = float(config['high_freq_cutoff_ratio'])
    if not 0 <= cutoff <= 1:
        raise ValueError('high_freq_cutoff_ratio must be between 0 and 1')
    return config


def generate_run_id():
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def get_git_commit_hash():
    try:
        result = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True)
        return result.strip()
    except Exception:
        return None


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_yaml(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def create_run_dir(config, outputs_base='outputs', run_id=None):
    run_id = run_id or generate_run_id()
    run_dir = Path(outputs_base) / run_id
    ensure_dir(run_dir / 'plots')
    ensure_dir(run_dir / 'checkpoints')
    snapshot = dict(config)
    snapshot['run_id'] = run_id
    snapshot['timestamp'] = snapshot.get('timestamp') or utc_timestamp()
    save_yaml(run_dir / 'config_snapshot.yaml', snapshot)
    return run_id, run_dir


def experiment_params(config, run_id, git_commit_hash=None):
    return {
        'run_id': run_id,
        'experiment_name': config['experiment_name'],
        'config_filename': config.get('config_filename'),
        'config_path': config.get('config_path'),
        'spectral_mode': config['spectral_mode'],
        'cutoff_ratio': float(config['high_freq_cutoff_ratio']),
        'image_size': int(config['image_size']),
        'batch_size': int(config['batch_size']),
        'epochs': int(config['epochs']),
        'learning_rate': float(config['learning_rate']),
        'latent_dim': int(config['latent_dim']),
        'activation': config.get('activation', 'relu'),
        'leaky_relu_slope': float(config.get('leaky_relu_slope', 0.1)),
        'gmm_k_range': config['gmm_k_range'],
        'gmm_selection_method': config.get('gmm_selection_method', 'elbow'),
        'threshold_percentile': float(config['threshold_percentile']),
        'score_alpha': float(config.get('score_alpha', 1.0)),
        'reconstruction_loss': config.get('reconstruction_loss', 'mse'),
        'reconstruction_target': config.get('reconstruction_target', 'all_channels'),
        'use_pca': bool(config.get('use_pca', False)),
        'pca_n_components': config.get('pca_n_components', 0.95),
        'test_subset': config.get('test_subset', 'all_pneumonia'),
        'timestamp': utc_timestamp(),
        'git_commit_hash': git_commit_hash or get_git_commit_hash(),
    }


def resolve_run_dir(config_path, outputs_base='outputs'):
    current_config = validate_config(load_config(config_path))
    config_filename = Path(config_path).name
    outputs_base = Path(outputs_base)
    if not outputs_base.exists():
        return None

    best_run = None
    best_ts = None
    for run_dir in outputs_base.iterdir():
        if not run_dir.is_dir():
            continue
        if not (run_dir / 'checkpoints' / 'ae.pt').exists():
            continue
        snapshot = run_dir / 'config_snapshot.yaml'
        if not snapshot.exists():
            continue
        try:
            data = yaml.safe_load(snapshot.read_text(encoding='utf-8'))
        except Exception:
            continue
        if data.get('config_filename') != config_filename:
            continue
        # These fields affect only GMM evaluation/decision policy, not the AE
        # checkpoint. Changing them should not force retraining the autoencoder.
        evaluation_only_keys = {
            'config_path',
            'gmm_k_range',
            'gmm_selection_method',
            'threshold_percentile',
            'score_alpha',
            'use_pca',
            'pca_n_components',
            'test_subset',
        }
        comparable_keys = [key for key in current_config.keys() if key not in evaluation_only_keys]
        if any(data.get(key) != current_config.get(key) for key in comparable_keys):
            continue
        timestamp = data.get('timestamp')
        if timestamp is None:
            # keep the first candidate if no timestamp exists
            if best_run is None:
                best_run = run_dir
            continue
        if best_ts is None or timestamp > best_ts:
            best_ts = timestamp
            best_run = run_dir
    return best_run
