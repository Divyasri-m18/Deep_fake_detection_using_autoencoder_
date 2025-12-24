# Deepfake Detector: Anomaly Detection Architecture

This document explains the **Corrected Architecture** for the Deepfake Video Detector.

## 🚫 Why the old model failed
The previous implementation attempted to use a mix of Classifier logic (probability 0-1) and Reconstruction logic, but often defaulted to hardcoded thresholds that didn't match the model's output scale.
- **Issue:** It treated the Autoencoder like a classifier.
- **Result:** It predicted "REAL" for everything because the error (e.g., 0.05) was always below the default threshold (0.5).

## ✅ The New Logic: Strict Anomaly Detection
We treat Deepfake Detection as a **One-Class Classification** problem.

### 1. Training (Normality Learning)
- The Autoencoder is trained **ONLY** on **REAL** videos.
- It learns to reconstruct "normal" human faces perfectly.
- It sees NO fake videos during training.

### 2. Statistical Calibration
- After training, we pass a validation set of REAL faces through the model.
- We measure the **Baseline Error Distribution** (Mean $\mu$ and Standard Deviation $\sigma$).
- We calculate a dynamic threshold:
  $$ Threshold = \mu + (k \times \sigma) $$
  *(Default k=2.5)*

### 3. Inference (Detection)
- **Input:** Unknown video.
- **Process:**
    1. Extract faces.
    2. Attempt to reconstruct them.
    3. Measure **Reconstruction Error (MSE)**.
- **Decision:**
    - If Error > Threshold $\rightarrow$ **FAKE** (Anomaly detected; model cannot reconstruct it).
    - If Error $\le$ Threshold $\rightarrow$ **REAL** (Model recognizes it as normal).

## 🚀 Usage

### 1. Training (Crucial Step)
You MUST train the model to generate the `model_stats.json` file.
```bash
python training/train_autoencoder.py
```

### 2. Detection (CLI)
```bash
# Standard Detection
python inference/detect_video.py path/to/video.mp4

# More Strict (Lower k)
python inference/detect_video.py path/to/video.mp4 --k 2.0

# Debug Mode (See stats)
python inference/detect_video.py path/to/video.mp4 --debug
```
