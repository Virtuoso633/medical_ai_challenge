"""
Model utilities for EF prediction.
Handles model loading, inference, and EF calculation using EchoNet-Dynamic architecture.
"""

import os
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import Tuple, Optional
import numpy as np


# Model configuration
MODEL_NAME = "r2plus1d_18"
NUM_FRAMES = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_path() -> str:
    """Get the path to the pretrained model weights."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(script_dir, "models", "r2plus1d_18_32_2_pretrained.pt")


def load_model(weights_path: Optional[str] = None) -> nn.Module:
    """
    Load the pretrained EchoNet-Dynamic model.
    
    The model is based on R(2+1)D-18 architecture, modified for
    regression (EF prediction) instead of classification.
    
    Args:
        weights_path: Path to pretrained weights. If None, uses default path.
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    if weights_path is None:
        weights_path = get_model_path()
    
    # Create R(2+1)D-18 base model
    model = video_models.r2plus1d_18(pretrained=False)
    
    # Modify the final layer for regression (single output for EF)
    # Original fc: Linear(512, 400) for Kinetics-400 classification
    # We replace with Linear(512, 1) for EF regression
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    # Load pretrained weights if available
    if os.path.exists(weights_path):
        print(f"Loading pretrained weights from: {weights_path}")
        # Note: weights_only=False is required for EchoNet checkpoint which contains numpy arrays
        # This is safe as we're loading from the official EchoNet-Dynamic release
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Try to load weights, handling potential mismatches
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Pretrained weights not found at {weights_path}")
        print("Model will use random initialization. Results will be unreliable.")
        print("\nTo download pretrained weights, run:")
        print("  mkdir -p models")
        print("  wget -O models/r2plus1d_18_32_2_pretrained.pt \\")
        print("    https://github.com/echonet/dynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt")
    
    model = model.to(DEVICE)
    model.eval()
    
    return model


def predict_ef(model: nn.Module, video_tensor: np.ndarray) -> Tuple[float, dict]:
    """
    Predict Ejection Fraction from preprocessed video tensor.
    
    Args:
        model: Loaded PyTorch model
        video_tensor: Preprocessed video tensor [1, C, T, H, W]
        
    Returns:
        Tuple of (predicted_ef, additional_info_dict)
    """
    # Convert numpy array to PyTorch tensor
    if isinstance(video_tensor, np.ndarray):
        video_tensor = torch.from_numpy(video_tensor).float()
    
    video_tensor = video_tensor.to(DEVICE)
    
    with torch.no_grad():
        # Run inference
        output = model(video_tensor)
        
        # Get prediction (clamp to valid EF range 0-100%)
        ef_prediction = output.item()
        ef_prediction = max(0.0, min(100.0, ef_prediction))
    
    additional_info = {
        'raw_output': output.item(),
        'device': str(DEVICE),
        'model_name': MODEL_NAME
    }
    
    return ef_prediction, additional_info


def get_ef_category(ef: float) -> str:
    """
    Categorize EF value based on clinical guidelines.
    
    Args:
        ef: Ejection fraction percentage
        
    Returns:
        Clinical category string
    """
    if ef >= 55:
        return "Normal (≥55%)"
    elif ef >= 40:
        return "Mildly Reduced (40-54%)"
    elif ef >= 30:
        return "Moderately Reduced (30-39%)"
    else:
        return "Severely Reduced (<30%)"


def download_weights(output_path: Optional[str] = None) -> bool:
    """
    Download pretrained model weights from GitHub releases.
    
    Args:
        output_path: Path to save the weights. If None, uses default path.
        
    Returns:
        True if download successful, False otherwise
    """
    import urllib.request
    
    if output_path is None:
        output_path = get_model_path()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    url = "https://github.com/echonet/dynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt"
    
    print(f"Downloading pretrained weights from: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False
