import cv2
import os
import glob
from typing import List, Generator

def get_video_paths(directory: str, extensions: List[str] = ['.mp4', '.avi', '.mov']) -> List[str]:
    """
    Get all video file paths from a directory with given extensions.
    """
    video_paths = []
    for ext in extensions:
        # Recursive search can be enabled if needed, here just flat
        video_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return video_paths

def load_video_frames_generator(video_path: str, frame_interval: int = 1) -> Generator:
    """
    Yields frames from a video file at a specified interval.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb
        
        frame_count += 1
    
    cap.release()

def get_video_info(video_path: str):
    """
    Returns metadata about the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return info
