# Chest X-ray Anomaly Detection

This project implements anomaly detection in chest X-ray images using a Convolutional Autoencoder trained on normal images, combined with Gaussian Mixture Model (GMM) on latent vectors.

## Algorithm Overview

1. **Preprocessing**:
   - Load grayscale chest X-ray images.
   - Resize to 128x128.
   - Normalize to [0,1].
   - Compute 2D FFT log magnitude spectrum, normalize per image.
   - Create 2-channel input: [original image, log spectrum].

2. **Autoencoder Training**:
   - Train convolutional AE only on train/NORMAL images.
   - Loss: MSE reconstruction loss.

3. **Latent Vector Extraction**:
   - Extract latent vectors z from encoder for train/NORMAL.
   - Standardize z using StandardScaler.

4. **GMM Fitting**:
   - Fit GaussianMixture(n_components=1, covariance_type="full") on standardized z.

5. **Anomaly Scoring**:
   - For test images: compute z, standardize, GMM score = -gmm.score_samples(z).
   - Reconstruction error = MSE between input and reconstruction.
   - Normalize both scores to [0,1].
   - Final score = 0.5 * normalized_GMM_score + 0.5 * normalized_reconstruction_error.

6. **Thresholding**:
   - Compute GMM anomaly scores on test/NORMAL.
   - Threshold = 95th percentile of test NORMAL GMM scores.

7. **Evaluation**:
   - Classify as anomaly if final_score > threshold.
   - Compute confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC.
   - Plot score histograms with threshold.
   - Save example images: original, log spectrum, reconstruction, error maps.

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Dataset Structure

Place the dataset in `data/chest_xray/` with the following structure:
```
data/chest_xray/
  train/
    NORMAL/
      *.png
  test/
    NORMAL/
      *.png
    PNEUMONIA/
      *.png
```

## Usage

1. Train the Autoencoder:
   ```
   python src/train_ae.py --data_dir data/chest_xray --epochs 20 --batch_size 32
   ```

2. Evaluate anomaly detection:
   ```
   python src/evaluate_gmm.py --data_dir data/chest_xray
   ```

## Outputs

- `checkpoints/ae.pt`: Trained Autoencoder.
- `checkpoints/gmm_latent.joblib`: Scaler and GMM model.
- `outputs/score_histogram.png`: Score distribution histogram.
- `outputs/example_*.png`: Example images with reconstructions and errors.