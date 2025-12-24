import os
import sys
import time
from inference.detect_video import AnomalyDetector

class VideoPredictor:
    def __init__(self, config_path=None):
        try:
             self.detector = AnomalyDetector()
             self.model_loaded = True
        except Exception as e:
             print(f"Error initializing AnomalyDetector: {e}")
             self.model_loaded = False

    def predict(self, video_path, threshold=None):
        start_time = time.time()
        
        if not self.model_loaded:
             return self._get_error_response("Model not loaded")

        try:
            # Delegate to AnomalyDetector
            # Note: AnomalyDetector.detect handles the threshold logic internally based on stats
            # But if a manual threshold is passed via UI, we might want to respect it OR ignore it
            # The new logic prefers statistical threshold. passing k=manual?
            # For now, let's use the detector's logic but if threshold is passed (float), we might override k?
            # Actually, the UI slider provides a raw float (e.g., 0.05).
            # AnomalyDetector.detect calculates its own threshold (e.g. 0.07).
            # If the user provides a manual threshold in UI, we should probably compare mean_error against THAT.
            
            # Run detection (default config)
            result = self.detector.detect(video_path)
            
            # If UI provided a specific threshold, re-evaluate decision
            # (Allows user to play with slider)
            processing_time = time.time() - start_time
            
            final_threshold = result['threshold']
            is_fake = result['is_fake']
            prediction = result['prediction']
            
            if threshold is not None and isinstance(threshold, float) and threshold > 0:
                final_threshold = threshold
                is_fake = result['mean_error'] > final_threshold
                prediction = "FAKE" if is_fake else "REAL"

            # Calculate Confidence based on Final Threshold
            mean_error = result['mean_error']
            
            # Cap max expected error for normalization (e.g. 0.1 is very high for MSE)
            max_expected_error = 0.15 
            
            if is_fake:
                # Range: [threshold, max_expected_error] -> [50, 100]
                # If error > max, confidence = 100
                diff = mean_error - final_threshold
                r = max_expected_error - final_threshold
                if r <= 0: r = 0.001 # prevent div by zero
                score = 50 + (diff / r) * 50
                confidence = min(99.9, max(50.1, score))
            else:
                # Range: [0, threshold] -> [100, 50] (High confidence being Real = low error)
                # Ideally, if error is 0, confidence is 100% (Real).
                # If error is threshold, confidence is 50%.
                # We want "Confidence in Prediction". so if Real, 100% means definitely real.
                ratio = mean_error / final_threshold if final_threshold > 0 else 0
                score = 50 + (1.0 - ratio) * 50
                confidence = min(99.9, max(50.1, score))

            return {
                "prediction": prediction,
                "confidence": confidence,
                "mean_error": mean_error,
                "threshold": final_threshold,
                "cls_stats": {
                    "prob": result.get('cls_avg_prob', 0.0),
                    "votes": result.get('cls_vote_ratio', 0.0)
                },
                "stats": {
                    "frames_analyzed": result.get("total_frames", 0),
                    "faces_detected": result.get("total_faces", 0),
                    "time": processing_time 
                },
                "device": result.get("device", "CPU"),
                "visuals": result.get("visuals", []),
                "frame_errors": result.get("frame_errors", [])
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._get_error_response(str(e))

    def _get_error_response(self, msg):
        return {
                "prediction": "ERROR",
                "threshold": 0.0, # Replaced 'threshold' with 0.0 as it's not defined in this scope.
                "frame_errors": [],
                "cls_stats": {"prob": 0.0, "votes": 0.0},
                "stats": { "frames_analyzed": 0, "faces_detected": 0, "time": 0.0, "device": "N/A" }, # Replaced 'str(self.device)' with "N/A" as 'self.device' is not defined in this class.
                "visuals": []
            }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = VideoPredictor()
        print(p.predict(sys.argv[1]))
