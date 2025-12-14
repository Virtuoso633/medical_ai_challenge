"""
Explainability utilities using Grad-CAM for visualizing model predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Tuple, Optional
import io


class XRayExplainer:
    """Generates Grad-CAM visualizations for X-ray predictions."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize explainer with model.
        
        Args:
            model: torchxrayvision model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Identify target layer for Grad-CAM
        # For DenseNet models, use the last conv layer
        self.target_layer = self._get_target_layer()
        
    def _get_target_layer(self):
        """Get the appropriate layer for Grad-CAM visualization."""
        # For torchxrayvision DenseNet models
        if hasattr(self.model, 'features'):
            # DenseNet architecture
            if hasattr(self.model.features, 'denseblock4'):
                return [self.model.features.denseblock4.denselayer16.conv2]
            elif hasattr(self.model.features, 'norm5'):
                return [self.model.features.norm5]
        
        # Fallback to last conv layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
        
        raise ValueError("Could not find suitable layer for Grad-CAM")
    
    def generate_gradcam(
        self, 
        img_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_tensor: Preprocessed image tensor [1, 1, H, W]
            target_class: Index of class to visualize (None for highest prediction)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        img_tensor = img_tensor.to(self.device)
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Initialize Grad-CAM
        cam = GradCAM(model=self.model, target_layers=self.target_layer)
        
        # Generate CAM
        # Reshape input if needed (remove normalization for visualization)
        targets = None  # Will use the predicted class
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        
        # Get the first image's CAM
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam
    
    def create_overlay(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Create overlay of heatmap on original image.
        
        Args:
            original_image: Original PIL Image (grayscale)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay (0-1)
            
        Returns:
            PIL Image with heatmap overlay
        """
        # Resize heatmap to match original image size
        heatmap_resized = Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize(original_image.size, Image.BILINEAR)
        heatmap_resized = np.array(heatmap_resized) / 255.0
        
        # Convert grayscale image to RGB
        original_rgb = np.array(original_image.convert('RGB')) / 255.0
        
        # Apply colormap to heatmap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # RGB only
        
        # Create overlay
        overlay = (1 - alpha) * original_rgb + alpha * heatmap_colored
        overlay = (overlay * 255).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def create_visualization(
        self,
        img_tensor: torch.Tensor,
        original_image: Image.Image,
        predictions: dict,
        top_k: int = 3
    ) -> Tuple[Image.Image, dict]:
        """
        Create comprehensive visualization with multiple heatmaps.
        
        Args:
            img_tensor: Preprocessed image tensor
            original_image: Original PIL Image
            predictions: Dictionary of predictions {class_name: probability}
            top_k: Number of top predictions to visualize
            
        Returns:
            Tuple of (side-by-side image, heatmap metadata)
        """
        # Get top predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_preds = sorted_preds[:top_k]
        
        # Generate Grad-CAM for top prediction
        top_class_idx = list(predictions.keys()).index(top_preds[0][0])
        heatmap = self.generate_gradcam(img_tensor, target_class=top_class_idx)
        
        # Create overlay
        overlay_image = self.create_overlay(original_image, heatmap, alpha=0.4)
        
        # Create side-by-side visualization
        width, height = original_image.size
        combined = Image.new('RGB', (width * 2, height))
        combined.paste(original_image.convert('RGB'), (0, 0))
        combined.paste(overlay_image, (width, 0))
        
        metadata = {
            'visualized_class': top_preds[0][0],
            'visualized_probability': top_preds[0][1],
            'top_predictions': top_preds
        }
        
        return combined, metadata
    
    def create_individual_heatmap(
        self,
        img_tensor: torch.Tensor,
        original_image: Image.Image,
        class_idx: int,
        class_name: str
    ) -> Image.Image:
        """
        Create heatmap for a specific class.
        
        Args:
            img_tensor: Preprocessed image tensor
            original_image: Original PIL Image
            class_idx: Index of the class
            class_name: Name of the class
            
        Returns:
            PIL Image with heatmap overlay and label
        """
        heatmap = self.generate_gradcam(img_tensor, target_class=class_idx)
        overlay = self.create_overlay(original_image, heatmap, alpha=0.4)
        
        return overlay
