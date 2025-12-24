import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', conf_threshold=0.9):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device, thresholds=[0.6, 0.7, conf_threshold])
        print(f"MTCNN initialized on {self.device}")

    def detect_faces(self, frame):
        """
        Detects faces in a frame.
        Args:
            frame: numpy array (RGB) or PIL Image
        Returns:
            boxes: list of bounding boxes [x1, y1, x2, y2]
            probs: list of probabilities
        """
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        boxes, probs = self.mtcnn.detect(frame)
        return boxes, probs

    def extract_face(self, frame, box, margin=0):
        """
        Crops the face from the frame using the bounding box.
        """
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Add margin
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face = frame[y1:y2, x1:x2]
        return face

if __name__ == "__main__":
    # Test stub
    detector = FaceDetector(device='cpu')
    print("FaceDetector ready.")
