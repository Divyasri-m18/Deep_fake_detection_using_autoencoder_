# Deepfake Detector: Hybrid Architecture

## Overview
This project implements a **Hybrid Detection Pipeline** that combines two powerful approaches to robustly detect Deepfake videos:
1.  **Anomaly Detection (Autoencoder)**: Flags inputs that deviate from the statistical properties of "Real" faces.
2.  **Supervised Classification (CNN)**: Explicitly identifies known deepfake artifacts (e.g., blurring, blending boundaries).

This combination ensures high recall (catching weird anomalies) and high precision (catching known fakes).

## Components

### 1. Autoencoder (Stage 1)
- **Role**: Anomaly Detector.
- **Training**: Trained **ONLY** on Real faces.
- **Logic**: It learns to reconstruct real faces perfectly. When it sees a fake face, the reconstruction error (MSE) is high.
- **Metric**: Threshold = `Mean_Real_Error + (k * Std_Dev)` (Default k=2.5).

### 2. CNN Classifier (Stage 2)
- **Role**: Pattern Matcher.
- **Training**: Trained on **BOTH** Real and Fake faces.
- **Logic**: Learns specific features of deepfakes.
- **Metric**: Probability Score (0.0 - 1.0).

## Hybrid Decision Logic
A video is classified as **FAKE** if:
1.  It is statistically **Anomalous** (Error > Threshold).
2.  **OR** the Classifier sees a **Fake Pattern** (Prob > 0.5).

This "OR" logic acts as a safety net:
- If the fake is "perfect" (low anomaly), the Classifier might still catch a subtle artifact.
- If the fake is "unseen" (classifier fails), the Autoencoder will flag it as an anomaly.

## Usage

### Training
1.  **Train Anomaly Detector**:
    ```bash
    python training/train_autoencoder.py
    ```
2.  **Train Classifier**:
    ```bash
    python training/train_classifier.py
    ```

### Inference
Run detection on a video:
```bash
python inference/detect_video.py path/to/video.mp4 --k 2.5
```

### Debug Mode
Use `--debug` to see the internal decision engine:
```bash
python inference/detect_video.py path/to/video.mp4 --debug
```
**Output codes:**
- `Anomaly Found`: Autoencoder reconstruction error exceeded the statistical limit.
- `Classifier Pattern Match`: CNN detected fake features.
