"""
Enhanced 5-Model Ensemble Predictor for Chest X-Ray Pathology Detection.

Uses multiple torchxrayvision models with voting ensemble for improved accuracy.
Target: AUC 0.88-0.92 (up from 0.75-0.85 with 2 models)

Available models:
- densenet121-res224-all (trained on all datasets)
- densenet121-res224-nih (NIH ChestX-ray14)
- densenet121-res224-chex (CheXpert)
- densenet121-res224-mimic_ch (MIMIC-CXR)
- densenet121-res224-pc (PadChest)
"""

import time
import torch
import torchxrayvision as xrv
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    pathology: str
    probability: float
    confidence: float  # Agreement between models (0-1)
    individual_predictions: List[float]
    threshold: float
    above_threshold: bool


class EnhancedEnsemblePredictor:
    """
    5-Model Ensemble Predictor with voting and confidence estimation.
    
    Strategy:
    1. Load 5 diverse DenseNet models trained on different datasets
    2. Run inference on all models
    3. Average predictions (weighted by model reliability)
    4. Calculate confidence from model agreement
    5. Use optimized per-pathology thresholds
    """
    
    # Model configurations with their strengths
    MODEL_CONFIGS = [
        {
            'weights': 'densenet121-res224-all',
            'name': 'All Datasets',
            'weight': 1.2,  # Higher weight - most comprehensive
            'description': 'Trained on ALL available datasets'
        },
        {
            'weights': 'densenet121-res224-nih',
            'name': 'NIH ChestX-ray14',
            'weight': 1.0,
            'description': 'NIH dataset - 112K images, 14 labels'
        },
        {
            'weights': 'densenet121-res224-chex',
            'name': 'CheXpert',
            'weight': 1.1,  # Stanford dataset - high quality
            'description': 'Stanford CheXpert - 224K images'
        },
        {
            'weights': 'densenet121-res224-mimic_ch',
            'name': 'MIMIC-CXR',
            'weight': 1.0,
            'description': 'MIT MIMIC-CXR - 377K images'
        },
        {
            'weights': 'densenet121-res224-pc',
            'name': 'PadChest',
            'weight': 0.9,  # Slightly lower - Spanish dataset
            'description': 'PadChest - 160K images'
        },
    ]
    
    def __init__(self, device: Optional[str] = None, num_models: int = 5):
        """
        Initialize the ensemble predictor.
        
        Args:
            device: 'cuda' or 'cpu' (auto-detected if None)
            num_models: Number of models to load (1-5)
        """
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.num_models = min(num_models, len(self.MODEL_CONFIGS))
        self.models = []
        self.model_weights = []
        self.pathology_labels = None
        self.optimized_thresholds = None
        self._is_loaded = False
    
    def load(self) -> 'EnhancedEnsemblePredictor':
        """Load all ensemble models."""
        if self._is_loaded:
            return self
        
        print(f"Loading {self.num_models}-model ensemble on {self.device}...")
        start_time = time.time()
        
        # PyTorch 2.6+ compatibility
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        # Load each model
        for i, config in enumerate(self.MODEL_CONFIGS[:self.num_models]):
            print(f"  [{i+1}/{self.num_models}] Loading {config['name']}...")
            
            model = xrv.models.DenseNet(weights=config['weights'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            self.model_weights.append(config['weight'])
        
        # Restore original torch.load
        torch.load = original_load
        
        # Get pathology labels from first model
        self.pathology_labels = list(self.models[0].pathologies)
        
        # Initialize with default thresholds
        self.optimized_thresholds = self._get_default_thresholds()
        
        self._is_loaded = True
        load_time = time.time() - start_time
        
        print(f"\n✅ Ensemble loaded in {load_time:.2f}s")
        print(f"   Models: {self.num_models}")
        print(f"   Pathologies: {len(self.pathology_labels)}")
        print(f"   Device: {self.device}")
        
        return self
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get optimized thresholds per pathology (from model's op_threshs)."""
        thresholds = {}
        base_thresholds = self.models[0].op_threshs.cpu().numpy()
        
        for i, label in enumerate(self.pathology_labels):
            thresholds[label] = float(base_thresholds[i])
        
        return thresholds
    
    def predict(self, img_tensor: torch.Tensor) -> Dict[str, EnsemblePrediction]:
        """
        Run ensemble prediction on an image.
        
        Args:
            img_tensor: Preprocessed image tensor [1, 1, 224, 224]
            
        Returns:
            Dictionary of pathology -> EnsemblePrediction
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        
        img_tensor = img_tensor.to(self.device)
        
        # Collect predictions from all models
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(img_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
                all_predictions.append(probs)
        
        all_predictions = np.array(all_predictions)  # [num_models, num_pathologies]
        
        # Compute weighted ensemble predictions
        weights = np.array(self.model_weights[:self.num_models])
        weights = weights / weights.sum()  # Normalize
        
        weighted_avg = np.average(all_predictions, axis=0, weights=weights)
        
        # Compute confidence (model agreement) using std deviation
        std_dev = np.std(all_predictions, axis=0)
        confidence = 1 - (std_dev * 2)  # Higher std = lower confidence
        confidence = np.clip(confidence, 0, 1)
        
        # Build results
        results = {}
        for i, label in enumerate(self.pathology_labels):
            threshold = self.optimized_thresholds.get(label, 0.5)
            
            results[label] = EnsemblePrediction(
                pathology=label,
                probability=float(weighted_avg[i]),
                confidence=float(confidence[i]),
                individual_predictions=[float(p[i]) for p in all_predictions],
                threshold=threshold,
                above_threshold=weighted_avg[i] >= threshold
            )
        
        return results
    
    # def predict_top_k(
    #     self, 
    #     img_tensor: torch.Tensor, 
    #     k: int = 5
    # ) -> List[EnsemblePrediction]:
    #     """Get top K predictions sorted by probability."""
    #     all_preds = self.predict(img_tensor)
    #     sorted_preds = sorted(
    #         all_preds.values(), 
    #         key=lambda x: x.probability, 
    #         reverse=True
    #     )
    #     return sorted_preds[:k]
    
    # def predict_above_threshold(
    #     self, 
    #     img_tensor: torch.Tensor
    # ) -> List[EnsemblePrediction]:
    #     """Get only predictions above their thresholds."""
    #     all_preds = self.predict(img_tensor)
    #     above = [p for p in all_preds.values() if p.above_threshold]
    #     return sorted(above, key=lambda x: x.probability, reverse=True)
    
    # def get_model_agreement(self, img_tensor: torch.Tensor) -> Dict[str, float]:
    #     """
    #     Get agreement score for each pathology.
        
    #     Returns:
    #         Dictionary of pathology -> agreement score (0-1)
    #         Higher = models agree, Lower = models disagree
    #     """
    #     preds = self.predict(img_tensor)
    #     return {label: pred.confidence for label, pred in preds.items()}


# Convenience function
def create_ensemble(num_models: int = 5) -> EnhancedEnsemblePredictor:
    """Create and load an ensemble predictor."""
    predictor = EnhancedEnsemblePredictor(num_models=num_models)
    predictor.load()
    return predictor


if __name__ == "__main__":
    # Quick test
    print("Testing Enhanced Ensemble Predictor...")
    ensemble = create_ensemble(num_models=5)
    print(f"\nPathologies detected: {ensemble.pathology_labels}")
    print(f"Thresholds: {ensemble.optimized_thresholds}")
