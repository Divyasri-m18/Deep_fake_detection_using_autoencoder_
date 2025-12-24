import torch
import numpy as np
import sys
import os
from preprocessing.face_detection import FaceDetector
from models.autoencoder import Autoencoder
from inference.predict_video import VideoPredictor

def test_components():
    print("--- Testing Components ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Test Face Detector
    print("\n1. Testing FaceDetector...")
    try:
        detector = FaceDetector(device=device)
        # Create dummy image (black square with a white rectangle)
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8) 
        cv2_img = dummy_img.copy() # ensuring it's right format if needed
        boxes, _ = detector.detect_faces(dummy_img)
        print("FaceDetector ran. Boxes found:", boxes)
    except Exception as e:
        print(f"FaceDetector FAILED: {e}")

    # 2. Test Autoencoder Architecture
    print("\n2. Testing Autoencoder Model Structure...")
    try:
        model = Autoencoder().to(device)
        model.eval()
        
        # Inference on dummy tensor (N, 3, 128, 128)
        dummy_input = torch.randn(2, 3, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"Autoencoder Output Shape: {output.shape}")
        if output.shape == dummy_input.shape:
             print("Shape check PASSED (Input == Output).")
        else:
             print("Shape check FAILED.")
             
    except Exception as e:
        print(f"Autoencoder Inference FAILED: {e}")

    # 3. Test VideoPredictor Loader
    print("\n3. Testing VideoPredictor Initialization...")
    try:
        predictor = VideoPredictor()
        print(f"VideoPredictor initialized. Model Loaded: {predictor.model_loaded}")
        if not predictor.model_loaded:
            print("(Note: Model not found is expected if you haven't trained yet. Logic is working.)")
    except Exception as e:
        print(f"VideoPredictor Init FAILED: {e}")

if __name__ == "__main__":
    test_components()
