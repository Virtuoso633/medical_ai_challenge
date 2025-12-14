#!/usr/bin/env python3
"""
EF Inference Script - Echocardiography Ejection Fraction Prediction

This script predicts Left Ventricular Ejection Fraction (EF) from 
echocardiogram videos using a pretrained EchoNet-Dynamic model.

Usage:
    python ef_inference.py --video sample.avi --out results.json

Author: Based on EchoNet-Dynamic (Stanford)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.video_utils import validate_video, load_video, preprocess_frames, extract_key_frames
from utils.model_utils import load_model, predict_ef, get_ef_category, download_weights


def create_visualization(frames: np.ndarray, ef: float, output_dir: str, 
                         video_name: str, ed_idx: int = None, es_idx: int = None):
    """
    Create visualization of the prediction results.
    
    Args:
        frames: Original video frames [T, H, W, C]
        ef: Predicted ejection fraction
        output_dir: Directory to save visualizations
        video_name: Base name for output files
        ed_idx: End-diastolic frame index
        es_idx: End-systolic frame index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_frames = len(frames)
    
    # Estimate ED and ES frames if not provided
    # ED is typically at max volume (largest LV), ES at min volume
    # Simple heuristic: ED at ~25% and ES at ~75% of cardiac cycle
    if ed_idx is None:
        ed_idx = total_frames // 4
    if es_idx is None:
        es_idx = (3 * total_frames) // 4
    
    # Ensure indices are valid
    ed_idx = min(ed_idx, total_frames - 1)
    es_idx = min(es_idx, total_frames - 1)
    
    # Create figure with key frames
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample frames: beginning, ED, ES
    sample_indices = [0, ed_idx, es_idx]
    titles = ['First Frame', f'Estimated ED (Frame {ed_idx})', f'Estimated ES (Frame {es_idx})']
    
    for ax, idx, title in zip(axes, sample_indices, titles):
        ax.imshow(frames[idx])
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Add EF prediction as overall title
    category = get_ef_category(ef)
    fig.suptitle(f'Predicted EF: {ef:.1f}% - {category}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    vis_path = os.path.join(output_dir, f'{video_name}_visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {vis_path}")
    
    # Create a simple frame montage
    create_frame_montage(frames, ef, output_dir, video_name)
    
    return vis_path


def create_frame_montage(frames: np.ndarray, ef: float, output_dir: str, video_name: str):
    """
    Create a montage of sampled frames from the video.
    
    Args:
        frames: Original video frames
        ef: Predicted EF
        output_dir: Output directory
        video_name: Base name for output
    """
    # Sample 8 frames evenly across the video
    num_samples = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    
    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.imshow(frames[idx])
        ax.set_title(f'Frame {idx}', fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(num_samples, 8):
        axes[i].axis('off')
    
    category = get_ef_category(ef)
    fig.suptitle(f'Frame Montage | Predicted EF: {ef:.1f}% ({category})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    montage_path = os.path.join(output_dir, f'{video_name}_montage.png')
    plt.savefig(montage_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Frame montage saved to: {montage_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict Ejection Fraction from Echocardiogram Videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ef_inference.py --video echo.avi --out results.json
    python ef_inference.py --video echo.mp4 --out results.json --visualize
    python ef_inference.py --download-weights
        """
    )
    
    parser.add_argument('--video', type=str, 
                        help='Path to input echocardiogram video (AVI or MP4)')
    parser.add_argument('--out', type=str, default='results.json',
                        help='Path to output JSON file (default: results.json)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations (saved to out/ directory)')
    parser.add_argument('--output-dir', type=str, default='out',
                        help='Directory for visualization outputs (default: out)')
    parser.add_argument('--download-weights', action='store_true',
                        help='Download pretrained model weights')
    
    args = parser.parse_args()
    
    # Handle weight download
    if args.download_weights:
        success = download_weights(args.model)
        sys.exit(0 if success else 1)
    
    # Validate arguments
    if not args.video:
        parser.error("--video is required (unless using --download-weights)")
    
    print("=" * 60)
    print("EchoNet-Dynamic EF Prediction")
    print("=" * 60)
    
    # Step 1: Validate input video
    print("\n[1/4] Validating input video...")
    is_valid, message = validate_video(args.video)
    if not is_valid:
        print(f"ERROR: {message}")
        sys.exit(1)
    print(f"✓ {message}")
    
    # Step 2: Load video
    print("\n[2/4] Loading and preprocessing video...")
    preprocess_start = time.time()
    
    frames, metadata = load_video(args.video)
    print(f"  - Loaded {len(frames)} frames ({metadata['width']}x{metadata['height']})")
    print(f"  - Duration: {metadata['duration_sec']:.2f}s @ {metadata['fps']:.1f} FPS")
    
    video_tensor = preprocess_frames(frames)
    print(f"  - Preprocessed tensor shape: {video_tensor.shape}")
    
    preprocess_time = (time.time() - preprocess_start) * 1000  # Convert to ms
    print(f"  - Preprocessing time: {preprocess_time:.1f}ms")
    
    # Step 3: Load model and run inference
    print("\n[3/4] Running model inference...")
    
    model_load_start = time.time()
    model = load_model(args.model)
    model_load_time = (time.time() - model_load_start) * 1000
    print(f"  - Model loaded in {model_load_time:.1f}ms")
    
    inference_start = time.time()
    ef_prediction, additional_info = predict_ef(model, video_tensor)
    inference_time = (time.time() - inference_start) * 1000
    print(f"  - Inference time: {inference_time:.1f}ms")
    
    # Step 4: Generate output
    print("\n[4/4] Generating output...")
    
    category = get_ef_category(ef_prediction)
    
    results = {
        'input_video': os.path.abspath(args.video),
        'ejection_fraction': round(ef_prediction, 2),
        'ef_category': category,
        'timing': {
            'preprocessing_time_ms': round(preprocess_time, 2),
            'model_load_time_ms': round(model_load_time, 2),
            'inference_time_ms': round(inference_time, 2),
            'total_time_ms': round(preprocess_time + model_load_time + inference_time, 2)
        },
        'video_metadata': {
            'frame_count': metadata['frame_count'],
            'fps': metadata['fps'],
            'resolution': f"{metadata['width']}x{metadata['height']}",
            'duration_sec': round(metadata['duration_sec'], 2)
        },
        'model_info': {
            'name': additional_info['model_name'],
            'device': additional_info['device']
        }
    }
    
    # Save JSON output
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {args.out}")
    
    # Generate visualizations if requested
    if args.visualize:
        video_name = Path(args.video).stem
        vis_path = create_visualization(
            frames, ef_prediction, args.output_dir, video_name
        )
        results['visualization'] = vis_path
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Predicted EF: {ef_prediction:.1f}%")
    print(f"  Category: {category}")
    print(f"  Total Processing Time: {results['timing']['total_time_ms']:.1f}ms")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
