"""
Image preprocessing utilities for X-ray images.
Handles JPG, PNG, and DICOM formats with de-identification.
"""

import io
import pydicom
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional


class XRayPreprocessor:
    """Handles preprocessing of X-ray images for torchxrayvision models."""
    
    def __init__(self, target_size: int = 224):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (torchxrayvision uses 224x224)
        """
        self.target_size = target_size
        
    def validate_file(self, file_bytes: bytes, filename: str, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        
        # Check file extension
        ext = filename.lower().split('.')[-1]
        valid_extensions = ['jpg', 'jpeg', 'png', 'dcm', 'dicom']
        if ext not in valid_extensions:
            return False, f"Invalid file type. Allowed types: {', '.join(valid_extensions)}"
        
        return True, None
    
    def load_dicom(self, file_bytes: bytes) -> Tuple[np.ndarray, dict]:
        """
        Load DICOM file and strip metadata for de-identification.
        
        Args:
            file_bytes: DICOM file content
            
        Returns:
            Tuple of (image_array, safe_metadata)
        """
        # Load DICOM
        dicom = pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Extract pixel array
        image_array = dicom.pixel_array
        
        # Extract only safe, non-identifying metadata
        safe_metadata = {
            'modality': getattr(dicom, 'Modality', 'Unknown'),
            'rows': getattr(dicom, 'Rows', image_array.shape[0]),
            'columns': getattr(dicom, 'Columns', image_array.shape[1]),
            'bits_stored': getattr(dicom, 'BitsStored', None),
        }
        
        return image_array, safe_metadata
    
    def load_image(self, file_bytes: bytes, filename: str) -> Tuple[Image.Image, dict]:
        """
        Load image from bytes, handling both standard formats and DICOM.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (PIL Image, metadata dict)
        """
        ext = filename.lower().split('.')[-1]
        metadata = {'format': ext}
        
        if ext in ['dcm', 'dicom']:
            # Load DICOM and convert to PIL Image
            image_array, dicom_metadata = self.load_dicom(file_bytes)
            metadata.update(dicom_metadata)
            
            # Normalize to 8-bit for PIL
            if image_array.dtype != np.uint8:
                image_array = ((image_array - image_array.min()) / 
                              (image_array.max() - image_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array)
        else:
            # Load standard image formats
            image = Image.open(io.BytesIO(file_bytes))
            metadata['size'] = image.size
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        return image, metadata
    
    def preprocess_for_model(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL Image for torchxrayvision model.
        
        Args:
            image: PIL Image in grayscale
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # torchxrayvision expects images normalized to [0, 1] and resized
        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        # Apply transforms
        img_tensor = transform(image)
        
        # torchxrayvision models expect shape [1, 1, H, W] (batch, channels, height, width)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Normalize using torchxrayvision's normalization
        # Most models expect normalization with mean=0.5, std=0.5
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor
    
    def process_file(self, file_bytes: bytes, filename: str) -> Tuple[torch.Tensor, Image.Image, dict]:
        """
        Complete processing pipeline from file bytes to model-ready tensor.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (preprocessed_tensor, original_pil_image, metadata)
        """
        # Validate file
        is_valid, error = self.validate_file(file_bytes, filename)
        if not is_valid:
            raise ValueError(error)
        
        # Load and convert image
        image, metadata = self.load_image(file_bytes, filename)
        
        # Preprocess for model
        img_tensor = self.preprocess_for_model(image)
        
        return img_tensor, image, metadata
