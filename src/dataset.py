import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from utils_fft import compute_log_spectrum

class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray images with preprocessing.

    For each image:
    - Load as grayscale
    - Resize to 128x128
    - Normalize to [0,1]
    - Compute log spectrum
    - Create 2-channel input: [original, log_spectrum]

    Args:
        root_dir (str): Path to data/chest_xray
        split (str): 'train', 'test', or 'val'
    """
    def __init__(
        self,
        root_dir,
        split='train',
        image_size=128,
        spectral_mode='full_spectrum',
        high_freq_cutoff_ratio=0.25,
        include_pneumonia=True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = int(image_size)
        self.spectral_mode = spectral_mode
        self.high_freq_cutoff_ratio = float(high_freq_cutoff_ratio)
        self.include_pneumonia = include_pneumonia
        self.image_paths = []
        self.labels = []  # 0: NORMAL, 1: PNEUMONIA

        # Load NORMAL images
        normal_dir = os.path.join(root_dir, split, 'NORMAL')
        validation_filenames = set()
        if split == 'train':
            validation_dir = os.path.join(root_dir, 'val', 'NORMAL')
            if os.path.exists(validation_dir):
                validation_filenames = {
                    img_file
                    for img_file in os.listdir(validation_dir)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg'))
                }
        if os.path.exists(normal_dir):
            for img_file in os.listdir(normal_dir):
                if img_file in validation_filenames:
                    continue
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(normal_dir, img_file))
                    self.labels.append(0)

        # Load PNEUMONIA images (if exists)
        pneumonia_dir = os.path.join(root_dir, split, 'PNEUMONIA')
        if include_pneumonia and os.path.exists(pneumonia_dir):
            for img_file in os.listdir(pneumonia_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(pneumonia_dir, img_file))
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        # Resize from config so spectrum experiments are directly comparable.
        image = image.resize((self.image_size, self.image_size))
        # To numpy and normalize to [0,1]
        image = np.array(image, dtype=np.float32) / 255.0

        # Compute the configured spectral representation.
        log_spectrum = compute_log_spectrum(
            image,
            spectral_mode=self.spectral_mode,
            cutoff_ratio=self.high_freq_cutoff_ratio,
        )

        # Stack to 2 channels: channel 0 is the image, channel 1 is the FFT
        # spectrum. The reconstruction target is selected by the training config.
        x = np.stack([image, log_spectrum], axis=0)
        x = torch.from_numpy(x)

        return x, label
