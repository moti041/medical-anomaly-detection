import torch


def select_reconstruction_target(x_recon, x, target='all_channels'):
    if target == 'all_channels':
        return x_recon, x
    if target == 'fft_only':
        return x_recon[:, 1:2, :, :], x[:, 1:2, :, :]
    raise ValueError(f"Unsupported reconstruction_target '{target}'")


def reconstruction_error_per_sample(
    x_recon,
    x,
    mode='energy_normalized_mse',
    target='all_channels',
    eps=1e-8,
):
    x_recon_target, x_target = select_reconstruction_target(x_recon, x, target=target)
    mse = torch.mean((x_recon_target - x_target) ** 2, dim=(1, 2, 3))
    if mode == 'mse':
        return mse
    if mode == 'energy_normalized_mse':
        energy = torch.mean(x_target ** 2, dim=(1, 2, 3))
        return mse / (energy + eps)
    raise ValueError(f"Unsupported reconstruction_loss '{mode}'")


def reconstruction_loss(
    x_recon,
    x,
    mode='energy_normalized_mse',
    target='all_channels',
    eps=1e-8,
):
    return torch.mean(
        reconstruction_error_per_sample(
            x_recon,
            x,
            mode=mode,
            target=target,
            eps=eps,
        )
    )
