"""
CXR Foundation Predictor Wrapper.

This module provides a local wrapper that communicates with the CXR Foundation
model running on Google Colab via Gradio API.

Usage:
    1. Run the CXR Foundation Colab notebook and start the Gradio server
    2. Copy the public URL (ngrok/gradio public link)
    3. Initialize this predictor with the URL
    
Example:
    predictor = CXRFoundationPredictor(api_url="https://xxx.gradio.live")
    predictions = predictor.predict(image_bytes)
"""

import io
import base64
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from PIL import Image

try:
    from gradio_client import Client
    HAS_GRADIO_CLIENT = True
except ImportError:
    HAS_GRADIO_CLIENT = False

from .cxr_prompts import CXR_PROMPTS, get_prompts


@dataclass
class CXRFoundationPrediction:
    """Container for CXR Foundation prediction result."""
    pathology: str
    score: float  # Zero-shot similarity score
    positive_prompt: str
    negative_prompt: str


class CXRFoundationPredictor:
    """
    Wrapper for CXR Foundation model running on Colab.
    
    Connects to a Gradio API endpoint exposed from the Colab notebook
    and sends images for zero-shot classification.
    """
    
    def __init__(
        self, 
        api_url: Optional[str] = None,
        pathologies: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """
        Initialize the CXR Foundation predictor.
        
        Args:
            api_url: Gradio API URL from Colab (e.g., https://xxx.gradio.live)
            pathologies: List of pathologies to predict. Defaults to all.
            timeout: API call timeout in seconds
        """
        self.api_url = api_url
        self.pathologies = pathologies or list(CXR_PROMPTS.keys())
        self.timeout = timeout
        self._client = None
        self._is_connected = False
        
    def connect(self, api_url: Optional[str] = None) -> bool:
        """
        Connect to the CXR Foundation API.
        
        Args:
            api_url: Override the API URL
            
        Returns:
            True if connection successful
        """
        if not HAS_GRADIO_CLIENT:
            raise ImportError(
                "gradio_client is required. Install with: pip install gradio_client"
            )
        
        url = (api_url or self.api_url or "").strip()
        if not url:
            raise ValueError("API URL is required. Set api_url parameter.")
            
        try:
            self._client = Client(url)
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to CXR Foundation API: {e}")
            self._is_connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to the API."""
        return self._is_connected
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 for API transmission."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def predict_single(
        self, 
        image_bytes: bytes, 
        pathology: str
    ) -> CXRFoundationPrediction:
        """
        Get zero-shot prediction for a single pathology.
        
        Args:
            image_bytes: Raw image bytes (PNG/JPG)
            pathology: Pathology name to classify
            
        Returns:
            CXRFoundationPrediction with score
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")
            
        pos_prompt, neg_prompt = get_prompts(pathology)
        
        # Encode image
        image_b64 = self._encode_image(image_bytes)
        
        # Call API
        try:
            result = self._client.predict(
                image_b64,
                pos_prompt,
                neg_prompt,
                api_name="/predict"
            )
            score = float(result)
        except Exception as e:
            print(f"API error for {pathology}: {e}")
            score = 0.0
            
        return CXRFoundationPrediction(
            pathology=pathology,
            score=score,
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt
        )
    
    def predict(
        self, 
        image_bytes: bytes,
        pathologies: Optional[List[str]] = None
    ) -> Dict[str, CXRFoundationPrediction]:
        """
        Get zero-shot predictions for all specified pathologies.
        
        Args:
            image_bytes: Raw image bytes (PNG/JPG)
            pathologies: List of pathologies. Defaults to self.pathologies.
            
        Returns:
            Dict mapping pathology name to prediction
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")
            
        pathologies = pathologies or self.pathologies
        results = {}
        
        for pathology in pathologies:
            results[pathology] = self.predict_single(image_bytes, pathology)
            
        return results
    
    def predict_batch(
        self, 
        image_bytes: bytes
    ) -> Dict[str, CXRFoundationPrediction]:
        """
        Get predictions for all pathologies in a single API call.
        
        More efficient than predict() when the Colab notebook supports batch mode.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dict mapping pathology name to prediction
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to API. Call connect() first.")
            
        # Encode image
        image_b64 = self._encode_image(image_bytes)
        
        # Build prompts list
        prompts = []
        for pathology in self.pathologies:
            pos, neg = get_prompts(pathology)
            prompts.append({'pathology': pathology, 'pos': pos, 'neg': neg})
        
        try:
            # Call batch API
            result = self._client.predict(
                image_b64,
                prompts,
                api_name="/predict_batch"
            )
            
            # Parse results
            results = {}
            for item in result:
                results[item['pathology']] = CXRFoundationPrediction(
                    pathology=item['pathology'],
                    score=item['score'],
                    positive_prompt=item['pos'],
                    negative_prompt=item['neg']
                )
            return results
            
        except Exception as e:
            print(f"Batch API not available, falling back to individual calls: {e}")
            return self.predict(image_bytes)
    
    def get_top_predictions(
        self, 
        image_bytes: bytes, 
        k: int = 5
    ) -> List[CXRFoundationPrediction]:
        """Get top K predictions sorted by score."""
        predictions = self.predict(image_bytes)
        sorted_preds = sorted(
            predictions.values(), 
            key=lambda x: x.score, 
            reverse=True
        )
        return sorted_preds[:k]


class MockCXRFoundationPredictor(CXRFoundationPredictor):
    """
    Mock predictor for testing without a Colab connection.
    
    Returns random scores for testing the integration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_connected = True
        
    def connect(self, api_url: Optional[str] = None) -> bool:
        self._is_connected = True
        return True
    
    def predict_single(
        self, 
        image_bytes: bytes, 
        pathology: str
    ) -> CXRFoundationPrediction:
        import random
        pos_prompt, neg_prompt = get_prompts(pathology)
        
        return CXRFoundationPrediction(
            pathology=pathology,
            score=random.uniform(0.0, 1.0),
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt
        )


def create_predictor(
    api_url: Optional[str] = None, 
    mock: bool = False
) -> CXRFoundationPredictor:
    """
    Factory function to create a CXR Foundation predictor.
    
    Args:
        api_url: Gradio API URL from Colab
        mock: If True, create a mock predictor for testing
        
    Returns:
        CXRFoundationPredictor instance
    """
    if mock:
        predictor = MockCXRFoundationPredictor()
    else:
        predictor = CXRFoundationPredictor(api_url=api_url)
        
    return predictor
