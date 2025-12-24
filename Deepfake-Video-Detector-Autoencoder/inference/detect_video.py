import cv2
import torch
import numpy as np
import os
import sys
import yaml
import json
import argparse
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import Autoencoder
from models.classifier import DeepfakeClassifier
from preprocessing.face_detection import FaceDetector
from utils.video_utils import load_video_frames_generator
from utils.error_analysis import calculate_mse, get_threshold

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

class AnomalyDetector:
    def __init__(self, config_path=None):
        self.config = load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['training']['use_gpu'] else 'cpu')
        
        # --- Load Autoencoder (Anomaly) ---
        self.autoencoder = Autoencoder().to(self.device)
        ae_path = "models/trained/autoencoder.pth"
        if os.path.exists(ae_path):
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
            self.autoencoder.eval()
            print(f"Anomaly Model loaded from {ae_path}")
        else:
            raise FileNotFoundError(f"Autoencoder not found at {ae_path}")

        # --- Load Classifier (Supervised) ---
        self.classifier = DeepfakeClassifier(pretrained=False).to(self.device)
        cls_path = "models/trained/classifier.pth"
        if os.path.exists(cls_path):
            self.classifier.load_state_dict(torch.load(cls_path, map_location=self.device))
            self.classifier.eval()
            print(f"Classifier Model loaded from {cls_path}")
        else:
             print(f"Warning: Classifier not found at {cls_path}. Hybrid mode partial.")
             self.classifier = None
             
        # --- Load Stats ---
        stats_path = "models/trained/model_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Model stats loaded: Mean={self.stats['mean']:.4f}, Std={self.stats['std']:.4f}")
        else:
            print("Warning: Stats not found. Using defaults.")
            self.stats = {'mean': 0.02, 'std': 0.01}
            
        # --- Detector & Transform ---
        self.detector = FaceDetector(device=self.device, conf_threshold=self.config['preprocessing']['conf_threshold'])
        self.face_size = 128 # Autoencoder size
        self.cls_face_size = 224 # Classifier size (EfficientNet)
        
        # We need two transforms? Or resize inside?
        # Autoencoder needs 128, Classifier needs 224 usually.
        # Let's resize tensor on the fly or just use 224 for both? 
        # Autoencoder structure is fixed to 128 (flatten size). 
        # So we treat them separately.
        
        self.transform_ae = transforms.Compose([
            transforms.Resize((self.face_size, self.face_size)),
            transforms.ToTensor(),
        ])
        
        self.transform_cls = transforms.Compose([
            transforms.Resize((self.cls_face_size, self.cls_face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, video_path, k=2.5, percentile=None):
        frame_interval = self.config['preprocessing']['frame_interval']
        
        # Determine if input is video or image
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                img = cv2.imread(video_path)
                if img is None: raise ValueError("Could not read image.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames = [img] 
                print(f"Processing SINGLE image: {video_path}")
            except Exception as e:
                print(f"Error reading image: {e}")
                frames = []
        else:
            frames = load_video_frames_generator(video_path, frame_interval)
        
        video_ae_errors = []
        video_cls_probs = []
        visualization_samples = [] # Tuple: (ae_error, cls_prob, orig, recon)
        
        batch_faces_orig = []
        
        print("Scanning...", end="", flush=True)
        
        frame_count = 0
        total_faces = 0
        batch_frame_indices = []
        frame_max_errors = {} # frame_idx -> max_error
        
        print("Scanning...", end="", flush=True)
        
        for i, frame in enumerate(frames):
            frame_count += 1
            boxes, _ = self.detector.detect_faces(frame)
            if boxes is None: continue
            
            total_faces += len(boxes)
            
            for box in boxes:
                face = self.detector.extract_face(frame, box, margin=15)
                if face.size == 0: continue
                
                batch_faces_orig.append(face)
                batch_frame_indices.append(i)
            
            if len(batch_faces_orig) >= 16:
                # Track current errors length to slice new ones later
                prev_len = len(video_ae_errors)
                self._process_hybrid_batch(batch_faces_orig, video_ae_errors, video_cls_probs, visualization_samples)
                
                # Map new errors to frames
                new_errors = video_ae_errors[prev_len:]
                for idx, err in zip(batch_frame_indices, new_errors):
                    frame_max_errors[idx] = max(frame_max_errors.get(idx, 0.0), err)
                
                batch_faces_orig = []
                batch_frame_indices = []
                print(".", end="", flush=True)

        if batch_faces_orig:
            prev_len = len(video_ae_errors)
            self._process_hybrid_batch(batch_faces_orig, video_ae_errors, video_cls_probs, visualization_samples)
            
            new_errors = video_ae_errors[prev_len:]
            for idx, err in zip(batch_frame_indices, new_errors):
                frame_max_errors[idx] = max(frame_max_errors.get(idx, 0.0), err)
            
        print("Done.")
        
        if not video_ae_errors:
            return {
                "prediction": "UNKNOWN", "mean_error": 0.0, "threshold": 0.0, "is_fake": False, "cls_fake_prob": 0.0
            }

        # --- Decision Logic (Hybrid) ---
        
        # 1. Anomaly Decision
        if percentile:
            threshold = self.stats.get(f'percentile_{percentile}', self.stats['mean'] + 3*self.stats['std'])
        else:
            threshold = self.stats['mean'] + (k * self.stats['std'])
            
        mean_ae_error = np.mean(video_ae_errors)
        is_anomaly = mean_ae_error > threshold
        
        # 2. Classifier Decision
        if video_cls_probs and self.classifier:
            avg_cls_prob = np.mean(video_cls_probs)
            # Majority Vote
            fake_votes = sum(1 for p in video_cls_probs if p > 0.5)
            vote_ratio = fake_votes / len(video_cls_probs)
            is_classified_fake = vote_ratio > 0.5
        else:
            avg_cls_prob = 0.0
            is_classified_fake = False
            vote_ratio = 0.0

        # 3. Final Fusion
        # OR Logic: If it looks weird (Anomaly) OR looks essentially fake (Classifier) -> FAKE
        is_fake = is_anomaly or is_classified_fake
        prediction = "FAKE" if is_fake else "REAL"

        reason = []
        if is_anomaly: reason.append(f"Anomaly Found (Error {mean_ae_error:.4f} > {threshold:.4f})")
        if is_classified_fake: reason.append(f"Classifier Pattern Match (Confidence {avg_cls_prob:.1%}, Votes {vote_ratio:.1%})")
        if not is_fake: reason.append("No Anomalies or Fake Patterns Detected")
        
        print(f"\n--- Result: {prediction} ---")
        print(f"Reason: {', '.join(reason)}")
        
        return {
            "prediction": prediction,
            "mean_error": float(mean_ae_error),
            "threshold": float(threshold),
            "cls_avg_prob": float(avg_cls_prob),
            "cls_vote_ratio": float(vote_ratio),
            "is_fake": is_fake,
            "is_anomaly": bool(is_anomaly),
            "is_classified_fake": bool(is_classified_fake),
            "visuals": visualization_samples[:5],
            "total_frames": frame_count,
            "total_faces": total_faces,
            "frame_errors": [frame_max_errors.get(i, 0.0) for i in range(frame_count)],
            "device": str(self.device).upper()
        }

    def _process_hybrid_batch(self, batch_originals, ae_errors, cls_probs, visuals):
        if not batch_originals: return
        
        # Prepare Inputs
        ae_inputs = []
        cls_inputs = []
        
        for face in batch_originals:
            pil_face = Image.fromarray(face)
            ae_inputs.append(self.transform_ae(pil_face))
            if self.classifier:
                cls_inputs.append(self.transform_cls(pil_face))

        ae_t = torch.stack(ae_inputs).to(self.device)
        
        # Run Autoencoder
        with torch.no_grad():
            recon = self.autoencoder(ae_t)
            mse_batch = calculate_mse(ae_t, recon)
            ae_errors.extend(mse_batch)
            
            probs_batch = [0.0] * len(batch_originals)
            if self.classifier and cls_inputs:
                cls_t = torch.stack(cls_inputs).to(self.device)
                logits = self.classifier(cls_t)
                # Output is (N, 1) probability due to Sigmoid
                probs_batch = logits.squeeze(1).cpu().numpy().tolist()
                cls_probs.extend(probs_batch)
        
        # Store Visuals
        if len(visuals) < 5:
            for i in range(len(batch_originals)):
                if len(visuals) >= 5: break
                
                # Recon to numpy
                recon_np = recon[i].cpu().permute(1, 2, 0).numpy()
                recon_np = np.clip(recon_np * 255.0, 0, 255).astype(np.uint8)
                
                visuals.append({
                    "orig": batch_originals[i],
                    "recon": recon_np,
                    "error": ae_errors[-len(batch_originals)+i],
                    "prob": probs_batch[i] if self.classifier else 0.0
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to video or image")
    parser.add_argument("--k", type=float, default=2.5, help="Std deviation multiplier for threshold")
    parser.add_argument("--percentile", type=float, help="Use percentile threshold (e.g., 95, 99)")
    parser.add_argument("--debug", action="store_true", help="Show debug stats")
    
    args = parser.parse_args()
    
    detector = AnomalyDetector()
    res = detector.detect(args.video_path, k=args.k, percentile=args.percentile)
    
    if args.debug:
        print("\n[Debug Stats]")
        print(f"Mean Error: {res['mean_error']:.5f}")
        print(f"Threshold:  {res['threshold']:.5f}")
        print(f"Classifier Prob: {res['cls_avg_prob']:.4f}")
        print(f"Classifier Votes: {res['cls_vote_ratio']:.2%}")
