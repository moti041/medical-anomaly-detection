import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from experiment_utils import (
    create_run_dir,
    experiment_params,
    load_config,
    save_json,
    validate_config,
)
from losses import reconstruction_loss
from model import Autoencoder
from tracker_integration import log_run


DEFAULT_CONFIG = {
    'experiment_name': 'ae_full_spectrum',
    'spectral_mode': 'full_spectrum',
    'image_size': 128,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'latent_dim': 128,
    'gmm_k_range': [1, 6],
    'threshold_percentile': 95,
    'score_alpha': 1.0,
    'reconstruction_loss': 'energy_normalized_mse',
    'reconstruction_target': 'fft_only',
    'high_freq_cutoff_ratio': 0.25,
    'data_dir': 'data/chest_xray',
    'config_filename': 'cli_args',
    'config_path': None,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder on NORMAL chest X-rays')
    parser.add_argument('--config', type=str, help='Path to YAML experiment config')
    parser.add_argument('--data_dir', type=str, help='Path to data/chest_xray')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    return parser.parse_args()


def load_runtime_config(args):
    config = load_config(args.config) if args.config else dict(DEFAULT_CONFIG)
    config.setdefault('data_dir', 'data/chest_xray')
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    return validate_config(config)


def main():
    args = parse_args()
    config = load_runtime_config(args)
    data_dir = config['data_dir']

    expected_dirs = [
        os.path.join(data_dir, 'train', 'NORMAL'),
        os.path.join(data_dir, 'test', 'NORMAL'),
        os.path.join(data_dir, 'test', 'PNEUMONIA'),
    ]
    missing_dirs = [d for d in expected_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(f'Error: The following directories do not exist: {missing_dirs}')
        print(f'Expected structure under {data_dir}:')
        print('  train/NORMAL/')
        print('  test/NORMAL/')
        print('  test/PNEUMONIA/')
        return

    run_id, run_dir = create_run_dir(config)
    checkpoint_dir = run_dir / 'checkpoints'
    params = experiment_params(config, run_id)
    save_json(run_dir / 'params.json', params)

    train_dataset = ChestXrayDataset(
        data_dir,
        'train',
        image_size=config['image_size'],
        spectral_mode=config['spectral_mode'],
        high_freq_cutoff_ratio=config['high_freq_cutoff_ratio'],
        include_pneumonia=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)

    model = Autoencoder(latent_dim=int(config['latent_dim']), image_size=int(config['image_size']))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    reconstruction_loss_mode = config.get('reconstruction_loss', 'mse')
    reconstruction_target = config.get('reconstruction_target', 'all_channels')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_loss = float('inf')
    final_loss = float('inf')
    patience = 5
    patience_counter = 0
    epochs_ran = 0

    for epoch in range(int(config['epochs'])):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = reconstruction_loss(
                x_recon,
                x,
                mode=reconstruction_loss_mode,
                target=reconstruction_target,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(train_loader)
        epochs_ran = epoch + 1

        if final_loss < best_loss:
            best_loss = final_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'ae_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        print(f'Epoch {epoch + 1}/{config["epochs"]}, Loss: {final_loss:.4f}')

    model.load_state_dict(torch.load(checkpoint_dir / 'ae_best.pt', map_location=device))
    torch.save(model.state_dict(), checkpoint_dir / 'ae.pt')
    print(f'Model saved to {checkpoint_dir / "ae.pt"}')

    metrics = {
        'best_train_loss': float(best_loss),
        'final_train_loss': float(final_loss),
        'epochs_ran': int(epochs_ran),
        'total_train_samples': len(train_dataset),
        'reconstruction_loss': reconstruction_loss_mode,
        'reconstruction_target': reconstruction_target,
    }
    save_json(run_dir / 'metrics.json', metrics)

    log_run(
        experiment_name=config['experiment_name'],
        parameters=params,
        metrics=metrics,
        notes=(
            'Autoencoder training on NORMAL chest X-rays. The run directory contains '
            'a config snapshot so spectrum experiments can be reproduced exactly.'
        ),
        tags=['training', 'autoencoder', config['spectral_mode'], 'chest_xray'],
        artifacts=[
            str(Path(run_dir / 'config_snapshot.yaml')),
            str(Path(run_dir / 'params.json')),
            str(Path(run_dir / 'metrics.json')),
            str(Path(checkpoint_dir / 'ae.pt')),
            str(Path(checkpoint_dir / 'ae_best.pt')),
        ],
    )

    print(f'Run ID: {run_id}')
    print(f'Run outputs: {run_dir}')


if __name__ == '__main__':
    main()
