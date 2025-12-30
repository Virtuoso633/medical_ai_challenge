"""
Text prompts for CXR Foundation zero-shot classification.

Each pathology has a (positive_prompt, negative_prompt) pair used to compute
similarity scores between image embeddings and text embeddings.
"""

# Standard pathology prompts for zero-shot classification
# Format: 'PathologyName': ('positive_prompt', 'negative_prompt')
CXR_PROMPTS = {
    # Core pathologies (from torchxrayvision)
    'Atelectasis': ('atelectasis present', 'normal lung expansion'),
    'Cardiomegaly': ('enlarged heart cardiomegaly', 'normal heart size'),
    'Consolidation': ('lung consolidation', 'clear lung fields'),
    'Edema': ('pulmonary edema present', 'no pulmonary edema'),
    'Effusion': ('pleural effusion present', 'no pleural effusion'),
    'Emphysema': ('emphysema findings', 'normal lung parenchyma'),
    'Fibrosis': ('pulmonary fibrosis', 'normal lung texture'),
    'Hernia': ('hiatal hernia visible', 'no hernia'),
    'Infiltration': ('lung infiltration present', 'clear lung fields'),
    'Mass': ('lung mass present', 'no lung mass'),
    'Nodule': ('pulmonary nodule present', 'no pulmonary nodule'),
    'Pleural_Thickening': ('pleural thickening', 'normal pleura'),
    'Pneumonia': ('pneumonia findings', 'no pneumonia'),
    'Pneumothorax': ('pneumothorax present', 'no pneumothorax'),
    
    # Additional pathologies
    'Enlarged Cardiomediastinum': ('enlarged cardiomediastinum', 'normal mediastinum'),
    'Lung Opacity': ('lung opacity present', 'clear lungs'),
    'Lung Lesion': ('lung lesion visible', 'no lung lesion'),
    'Fracture': ('rib fracture present', 'no fracture'),
    'Support Devices': ('medical support devices present', 'no support devices'),
    'No Finding': ('normal chest x-ray', 'abnormal chest x-ray findings'),
}

# Alternative prompts that may perform better for specific conditions
CXR_PROMPTS_DETAILED = {
    'Atelectasis': (
        'subsegmental atelectasis with volume loss',
        'fully expanded lungs without atelectasis'
    ),
    'Cardiomegaly': (
        'cardiac silhouette enlarged greater than 50 percent of thoracic width',
        'normal cardiac silhouette less than 50 percent of thoracic width'
    ),
    'Consolidation': (
        'airspace consolidation with air bronchograms',
        'clear lung parenchyma without consolidation'
    ),
    'Effusion': (
        'pleural effusion with blunting of costophrenic angle',
        'sharp costophrenic angles without effusion'
    ),
    'Pneumothorax': (
        'pneumothorax with visible pleural line and absent lung markings',
        'normal lung markings without pneumothorax'
    ),
}


def get_prompts(pathology: str, detailed: bool = False) -> tuple:
    """
    Get positive and negative prompts for a pathology.
    
    Args:
        pathology: Name of the pathology
        detailed: Use detailed prompts if available
        
    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    if detailed and pathology in CXR_PROMPTS_DETAILED:
        return CXR_PROMPTS_DETAILED[pathology]
    return CXR_PROMPTS.get(pathology, (f'{pathology} present', f'no {pathology}'))


def get_all_pathologies() -> list:
    """Get list of all supported pathologies."""
    return list(CXR_PROMPTS.keys())
