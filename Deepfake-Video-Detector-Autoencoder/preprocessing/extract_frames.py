import os
import cv2
import sys
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import get_video_paths, load_video_frames_generator
from preprocessing.face_detection import FaceDetector

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def extract_faces_from_videos(video_dir, output_dir, detector, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_paths = get_video_paths(video_dir)
    frame_interval = config['preprocessing']['frame_interval']
    face_size = config['preprocessing']['face_size']
    
    print(f"Found {len(video_paths)} videos in {video_dir}")
    
    generated_count = 0
    
    for video_path in tqdm(video_paths, desc="Processing Videos"):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Use generator to save memory
        frame_gen = load_video_frames_generator(video_path, frame_interval)
        
        frame_idx = 0
        for frame in frame_gen:
            boxes, _ = detector.detect_faces(frame)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face_img = detector.extract_face(frame, box, margin=20)
                    if face_img.size == 0:
                        continue
                        
                    # Resize to target size here or in normalize step?
                    # Doing it here saves disk space
                    try:
                        face_img = cv2.resize(face_img, (face_size, face_size))
                        
                        # Save
                        save_name = f"{base_name}_frame{frame_idx}_face{i}.jpg"
                        save_path = os.path.join(output_dir, save_name)
                        
                        # Convert RGB back to BGR for OpenCV saving
                        cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        generated_count += 1
                    except Exception as e:
                        pass # fast fail on resize issues
            
            frame_idx += 1

    print(f"Extracted {generated_count} faces to {output_dir}")

def main():
    config = load_config()
    detector = FaceDetector(device='cuda' if torch.cuda.is_available() else 'cpu', 
                            conf_threshold=config['preprocessing']['conf_threshold'])
    
    # Process Real Videos
    print("Processing Real Videos...")
    extract_faces_from_videos(
        config['paths']['raw_real_videos'], 
        os.path.join(config['paths']['processed_faces'], 'real'),
        detector,
        config
    )
    
    # Process Fake Videos (Optional for training autoencoder, usually only real is needed for training)
    # But for testing classifier we might need both. 
    # The prompt says "Train ONLY on real faces".
    # So we strictly need real faces for training.
    # We might extract fake faces for evaluation though.
    
    print("Processing Fake Videos (for testing)...")
    extract_faces_from_videos(
        config['paths']['raw_fake_videos'], 
        os.path.join(config['paths']['processed_faces'], 'fake'),
        detector,
        config
    )

if __name__ == "__main__":
    import torch
    main()
