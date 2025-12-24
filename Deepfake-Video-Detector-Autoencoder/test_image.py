import sys
from inference.predict_video import VideoPredictor
import cv2
import numpy as np

def test_image_prediction():
    # Create a dummy image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Draw a face-like structure
    cv2.circle(img, (128, 128), 50, (200, 200, 200), -1) 
    cv2.imwrite("test_face.jpg", img)
    
    print("Created test_face.jpg")
    
    predictor = VideoPredictor()
    result = predictor.predict("test_face.jpg")
    
    print("Prediction Result:")
    print(result)
    
    if result['prediction'] in ['REAL', 'FAKE', 'UNKNOWN']:
        print("TEST PASSED: Image processed successfully.")
    else:
        print("TEST FAILED: Unexpected result.")

if __name__ == "__main__":
    test_image_prediction()
