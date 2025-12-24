# Deepfake Video Detector (EfficientNet + Autoencoder)

This project implements a high-performance Deepfake Video Detection system. 
It combines **EfficientNet-B0** (as a feature extractor, inspired by the DFDC winning solutions) with an **Autoencoder** anomaly detection head.

The model is designed to be trained on **Real** faces only. It learns to reconstruct real faces perfectly. When presented with a Fake/Deepfake face, the reconstruction error will be high.

## 📂 Project Structure
```
Deepfake-Video-Detector-Autoencoder/
├── data/                   # Data storage
│   ├── kaggle_140k/        # (NEW) Kaggle Dataset "140k-real-and-fake-faces"
│   ├── raw/                # Put your original videos here
│   │   ├── real/           # Real videos
│   │   └── fake/           # Fake videos (for testing)
│   └── processed/          # Extracted face crops
├── models/                 # PyTorch Models (EfficientNet-B0 Autoencoder)
├── training/               # Training scripts
├── inference/              # Prediction logic
├── app/                    # Streamlit Dashboard
├── configs/                # Configuration file
└── utils/                  # Helper functions
```

## 🚀 Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data (Recommended for High Accuracy)**
   - Download **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** from Kaggle.
   - Extract it into `data/kaggle_140k/`. 
   - The training script will automatically look for real faces there.

   *Alternatively, put your own **REAL** videos in `data/raw/real/` and run `python main.py preprocess`.*

## 🛠 Usage

### 1. Train Model
Trains the EfficientNet-Autoencoder. Prioritizes Kaggle data if found.
```bash
python main.py train
```
*Model saved to `models/trained/autoencoder.pth`*

### 2. Run Streamlit App (UI)
Launch the interactive web interface to upload and test videos.
```bash
python main.py ui
```
OR
```bash
streamlit run app/streamlit_app.py
```

### 5. CLI Single Video Prediction
```bash
python main.py predict --video path/to/video.mp4
```

## ⚙️ Configuration
You can adjust parameters in `configs/config.yaml`:
- `frame_interval`: How often to extraction frames (e.g., every 10th frame).
- `face_size`: Image size for the model (default 128x128).
- `batch_size`, `epochs`, `learning_rate`: Training hyperparameters.
- `mse_threshold`: Detection threshold (Manually tune this based on Evaluation results).

## 🧠 Methodology
- **Face Detection**: Uses MTCNN (via `facenet-pytorch`).
- **Model**: Autoencoder with Convolutional Encoder and Transpose Convolutional Decoder.
- **Detection**: 
  - $Error = MSE(Input, Reconstructed)$
  - If $MeanError > Threshold \rightarrow$ **FAKE**

## 📝 Notes
- The model assumes it is trained on faces similar to the test cases.
- For best results, train on a diverse dataset of real faces (e.g., DFDC, Celeb-DF).
