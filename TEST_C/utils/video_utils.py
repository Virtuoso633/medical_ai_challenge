"""
Video processing utilities for EF prediction.
Handles video loading, validation, and preprocessing for EchoNet-Dynamic model.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional


# Constants
MAX_FILE_SIZE_MB = 50
VALID_EXTENSIONS = {'.avi', '.mp4', '.AVI', '.MP4'}
TARGET_SIZE = (112, 112)  # EchoNet-Dynamic expected input size
NUM_FRAMES = 32  # Number of frames to sample
SAMPLING_PERIOD = 2  # Frame sampling period


def validate_video(video_path: str) -> Tuple[bool, str]:
    """
    Validate video file for processing.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return False, f"Video file not found: {video_path}"
    
    # Check file extension
    _, ext = os.path.splitext(video_path)
    if ext not in VALID_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Supported formats: AVI, MP4"
    
    # Check file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds limit of {MAX_FILE_SIZE_MB}MB"
    
    # Try to open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"Unable to open video file: {video_path}"
    
    # Check if video has frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return False, "Video file has no frames"
    
    cap.release()
    return True, f"Valid video file ({frame_count} frames, {file_size_mb:.1f}MB)"


def load_video(video_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load video frames from file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frames array [T, H, W, C], video metadata dict)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    metadata = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration_sec': frame_count / fps if fps > 0 else 0
    }
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames), metadata


def preprocess_frames(frames: np.ndarray, num_frames: int = NUM_FRAMES, 
                      sampling_period: int = SAMPLING_PERIOD) -> np.ndarray:
    """
    Preprocess video frames for EchoNet-Dynamic model.
    
    Args:
        frames: Input frames array [T, H, W, C]
        num_frames: Number of frames to sample (default: 32)
        sampling_period: Frame sampling period (default: 2)
        
    Returns:
        Preprocessed tensor [1, C, T, H, W] ready for model input
    """
    total_frames = len(frames)
    
    # Calculate required frames with sampling period
    required_frames = num_frames * sampling_period
    
    if total_frames < required_frames:
        # If not enough frames, repeat the last frame
        pad_count = required_frames - total_frames
        last_frame = frames[-1:].repeat(pad_count, axis=0)
        frames = np.concatenate([frames, last_frame], axis=0)
    
    # Sample frames with the given period
    # Start from a position that allows sampling the required number of frames
    start_idx = 0
    if total_frames > required_frames:
        # Start from middle for better coverage
        start_idx = (total_frames - required_frames) // 2
    
    sampled_indices = [start_idx + i * sampling_period for i in range(num_frames)]
    sampled_frames = frames[sampled_indices]
    
    # Resize frames to target size
    resized_frames = []
    for frame in sampled_frames:
        resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized)
    
    resized_frames = np.array(resized_frames)
    
    # Normalize to [0, 1] and then apply ImageNet normalization
    frames_normalized = resized_frames.astype(np.float32) / 255.0
    
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frames_normalized = (frames_normalized - mean) / std
    
    # Reshape to [1, C, T, H, W] (batch, channels, time, height, width)
    # Current shape: [T, H, W, C]
    frames_tensor = np.transpose(frames_normalized, (3, 0, 1, 2))  # [C, T, H, W]
    frames_tensor = np.expand_dims(frames_tensor, axis=0)  # [1, C, T, H, W]
    
    return frames_tensor


def extract_key_frames(frames: np.ndarray, ed_idx: Optional[int] = None, 
                       es_idx: Optional[int] = None) -> dict:
    """
    Extract key frames (end-diastolic and end-systolic) from video.
    
    Args:
        frames: Input frames array [T, H, W, C]
        ed_idx: End-diastolic frame index (optional)
        es_idx: End-systolic frame index (optional)
        
    Returns:
        Dictionary containing key frames and their indices
    """
    result = {
        'total_frames': len(frames),
        'ed_frame': None,
        'ed_index': ed_idx,
        'es_frame': None,
        'es_index': es_idx
    }
    
    if ed_idx is not None and 0 <= ed_idx < len(frames):
        result['ed_frame'] = frames[ed_idx]
    
    if es_idx is not None and 0 <= es_idx < len(frames):
        result['es_frame'] = frames[es_idx]
    
    return result
