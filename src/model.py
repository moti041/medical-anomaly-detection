import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for 2-channel input (image + log spectrum).

    Encoder: Conv layers down to the configured latent vector size.
    Decoder: Transpose Conv layers back to 2-channel output.
    """
    def __init__(self, latent_dim=128, image_size=128):
        super(Autoencoder, self).__init__()
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)

        # Encoder - deeper network for better feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),    # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Better than flattening
            nn.Flatten(),                                             # (256*4*4)
            nn.Linear(256 * 4 * 4, self.latent_dim),
            nn.Dropout(0.1)  # Prevent overfitting
        )

        # Decoder - improved with batch norm and better upsampling
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * 4 * 4),                 # (256*4*4)
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),                            # (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),     # (2, 64, 64)
            nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        """
        Forward pass: encode and decode.

        Args:
            x (torch.Tensor): Input of shape (batch, 2, 128, 128)

        Returns:
            torch.Tensor: Reconstruction of shape (batch, 2, 128, 128)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        """
        Encode input to latent vector.

        Args:
            x (torch.Tensor): Input of shape (batch, 2, 128, 128)

        Returns:
            torch.Tensor: Latent vector of shape (batch, latent_dim)
        """
        return self.encoder(x)
