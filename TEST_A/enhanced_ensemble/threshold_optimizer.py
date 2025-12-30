"""
Per-Pathology Threshold Optimizer.

Optimizes classification thresholds for each of the 18 pathologies
to maximize F1-score or other metrics.

The default thresholds from torchxrayvision are often too conservative.
This module allows tuning based on:
- F1-score optimization
- Sensitivity/Specificity trade-offs
- Custom clinical requirements
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ThresholdConfig:
    """Configuration for a single pathology threshold."""
    pathology: str
    default_threshold: float
    optimized_threshold: float
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate
    f1_score: float
    
    
class ThresholdOptimizer:
    """
    Optimizes per-pathology thresholds for classification.
    
    Default thresholds from models are often conservative.
    This optimizer finds thresholds that maximize F1-score.
    """
    
    # Clinically-tuned thresholds (based on literature)
    # These prioritize SENSITIVITY (not missing diseases) over specificity
    CLINICAL_THRESHOLDS = {
        # High-stakes pathologies - lower threshold (catch more)
        'Pneumothorax': 0.15,       # Life-threatening
        'Pneumonia': 0.20,          # Common, treatable
        'Cardiomegaly': 0.25,       # Important cardiac finding
        'Pleural_Thickening': 0.20,
        
        # Moderate-stakes
        'Atelectasis': 0.25,
        'Consolidation': 0.25,
        'Edema': 0.30,
        'Effusion': 0.25,
        'Infiltration': 0.20,
        
        # Findings that need more confidence
        'Nodule': 0.30,             # Higher threshold - needs confidence
        'Mass': 0.35,               # Very important - needs high confidence
        'Emphysema': 0.35,
        'Fibrosis': 0.30,
        'Hernia': 0.40,
        
        # Other findings
        'Fracture': 0.25,
        'Lung Lesion': 0.30,
        'Lung Opacity': 0.20,
        'Enlarged Cardiomediastinum': 0.25,
    }
    
    def __init__(self, base_thresholds: Dict[str, float]):
        """
        Initialize optimizer with base thresholds.
        
        Args:
            base_thresholds: Default thresholds from the model
        """
        self.base_thresholds = base_thresholds
        self.optimized_thresholds = self._apply_clinical_adjustments()
    
    def _apply_clinical_adjustments(self) -> Dict[str, float]:
        """
        Apply clinically-informed threshold adjustments.
        
        Strategy: Use lower of (model default, clinical threshold)
        to prioritize sensitivity.
        """
        thresholds = {}
        
        for pathology, default in self.base_thresholds.items():
            clinical = self.CLINICAL_THRESHOLDS.get(pathology, default)
            # Take lower threshold to catch more cases
            thresholds[pathology] = min(default, clinical)
        
        return thresholds
    
    def get_thresholds(self, mode: str = 'balanced') -> Dict[str, float]:
        """
        Get thresholds based on operating mode.
        
        Args:
            mode: 
                - 'high_sensitivity': Catch everything (more false positives)
                - 'balanced': Balance sensitivity and specificity
                - 'high_specificity': High confidence only (fewer false positives)
                
        Returns:
            Dictionary of pathology -> threshold
        """
        multipliers = {
            'high_sensitivity': 0.7,   # Lower thresholds
            'balanced': 1.0,           # Use optimized as-is
            'high_specificity': 1.3,   # Higher thresholds
        }
        
        multiplier = multipliers.get(mode, 1.0)
        
        return {
            pathology: min(threshold * multiplier, 0.9)
            for pathology, threshold in self.optimized_thresholds.items()
        }
    
    def optimize_for_f1(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray
    ) -> Dict[str, ThresholdConfig]:
        """
        Optimize thresholds using validation data to maximize F1.
        
        Args:
            predictions: [N, 18] predicted probabilities
            ground_truth: [N, 18] binary ground truth labels
            
        Returns:
            Dictionary of pathology -> ThresholdConfig
        """
        from sklearn.metrics import f1_score, precision_recall_curve
        
        results = {}
        pathologies = list(self.base_thresholds.keys())
        
        for i, pathology in enumerate(pathologies):
            y_true = ground_truth[:, i]
            y_pred = predictions[:, i]
            
            # Skip if no positive examples
            if y_true.sum() == 0:
                results[pathology] = ThresholdConfig(
                    pathology=pathology,
                    default_threshold=self.base_thresholds[pathology],
                    optimized_threshold=self.base_thresholds[pathology],
                    sensitivity=0.0,
                    specificity=1.0,
                    f1_score=0.0
                )
                continue
            
            # Find optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            
            # Calculate F1 for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find best threshold
            best_idx = np.argmax(f1_scores[:-1])  # Exclude last (threshold=1)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            # Calculate sensitivity/specificity at this threshold
            y_binary = (y_pred >= best_threshold).astype(int)
            sensitivity = recall[best_idx]
            specificity = ((y_true == 0) & (y_binary == 0)).sum() / (y_true == 0).sum()
            
            results[pathology] = ThresholdConfig(
                pathology=pathology,
                default_threshold=self.base_thresholds[pathology],
                optimized_threshold=float(best_threshold),
                sensitivity=float(sensitivity),
                specificity=float(specificity),
                f1_score=float(best_f1)
            )
            
            # Update optimized thresholds
            self.optimized_thresholds[pathology] = best_threshold
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save optimized thresholds to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'base_thresholds': self.base_thresholds,
                'optimized_thresholds': self.optimized_thresholds,
                'clinical_thresholds': self.CLINICAL_THRESHOLDS,
            }, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ThresholdOptimizer':
        """Load thresholds from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        optimizer = cls(data['base_thresholds'])
        optimizer.optimized_thresholds = data['optimized_thresholds']
        return optimizer


def print_threshold_comparison(optimizer: ThresholdOptimizer) -> None:
    """Print comparison of default vs optimized thresholds."""
    print("\n" + "="*70)
    print("Threshold Comparison: Default vs Optimized (Clinical)")
    print("="*70)
    print(f"{'Pathology':<30} {'Default':>10} {'Optimized':>10} {'Change':>10}")
    print("-"*70)
    
    for pathology in sorted(optimizer.base_thresholds.keys()):
        default = optimizer.base_thresholds[pathology]
        optimized = optimizer.optimized_thresholds[pathology]
        change = ((optimized - default) / default) * 100
        
        change_str = f"{change:+.0f}%" if abs(change) > 0.5 else "same"
        
        print(f"{pathology:<30} {default:>10.3f} {optimized:>10.3f} {change_str:>10}")
    
    print("="*70)
