"""
Flask + Gradio X-Ray Pathology Prediction Application
Uses torchxrayvision for inference with Grad-CAM explainability.
"""

import os
import time
import traceback
from io import BytesIO
import torch
import torchxrayvision as xrv
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import gradio as gr
from PIL import Image

from utils.preprocessing import XRayPreprocessor
from utils.explainability import XRayExplainer


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE_MB = 10
MODEL_NAME = 'densenet121-res224-all'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model (loaded at startup)
model = None
device = None
preprocessor = None
explainer = None
pathology_labels = None


def load_model():
    """Load torchxrayvision model at startup."""
    global model, device, preprocessor, explainer, pathology_labels
    
    print("Loading torchxrayvision model...")
    start_time = time.time()
    
    # PyTorch 2.6+ compatibility: Monkey patch torch.load for torchxrayvision
    # torchxrayvision saves whole model objects which require weights_only=False
    original_load = torch.load
    def patched_load(*args, **kwargs):
        # Force weights_only=False for torchxrayvision model loading
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = xrv.models.DenseNet(weights=MODEL_NAME)
    
    # Restore original torch.load
    torch.load = original_load
    model.to(device)
    model.eval()
    
    # Get pathology labels
    pathology_labels = model.pathologies
    
    # Initialize preprocessor and explainer
    preprocessor = XRayPreprocessor(target_size=224)
    explainer = XRayExplainer(model, device=device)
    
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f}s")
    print(f"Pathologies: {pathology_labels}")
    
    return model


def predict_xray(file_bytes: bytes, filename: str):
    """
    Main prediction function.
    
    Args:
        file_bytes: Image file as bytes
        filename: Original filename
        
    Returns:
        Dictionary with predictions, metrics, and visualizations
    """
    try:
        # Track timing
        timings = {}
        
        # Preprocess
        preprocess_start = time.time()
        img_tensor, original_image, metadata = preprocessor.process_file(file_bytes, filename)
        img_tensor = img_tensor.to(device)
        timings['preprocessing_ms'] = round((time.time() - preprocess_start) * 1000, 2)
        
        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]
        timings['inference_ms'] = round((time.time() - inference_start) * 1000, 2)
        
        # Create predictions dictionary
        pred_dict = {label: float(prob) for label, prob in zip(pathology_labels, predictions)}
        
        # Sort by probability
        sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Generate Grad-CAM visualization
        gradcam_start = time.time()
        visualization, vis_metadata = explainer.create_visualization(
            img_tensor, original_image, pred_dict, top_k=3
        )
        timings['gradcam_ms'] = round((time.time() - gradcam_start) * 1000, 2)
        
        # Create results DataFrame for Gradio display
        results_df = pd.DataFrame([
            {
                'Pathology': label,
                'Probability': f"{prob:.4f}",
                'Percentage': f"{prob*100:.2f}%"
            }
            for label, prob in sorted_preds[:10]  # Top 10
        ])
        
        # Prepare metrics text
        metrics_text = f"""
**Performance Metrics:**
- Preprocessing: {timings['preprocessing_ms']}ms
- Inference: {timings['inference_ms']}ms
- Grad-CAM: {timings['gradcam_ms']}ms
- Total: {sum(timings.values())}ms

**Image Info:**
- Format: {metadata.get('format', 'Unknown')}
- Size: {metadata.get('size', 'N/A')}
- Modality: {metadata.get('modality', 'N/A')}

**Top Prediction:**
- {vis_metadata['visualized_class']}: {vis_metadata['visualized_probability']*100:.2f}%
"""
        
        return results_df, visualization, metrics_text, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, error_msg


def gradio_predict(file):
    """
    Gradio interface prediction function.
    
    Args:
        file: Gradio file upload object
        
    Returns:
        Tuple of (results_dataframe, visualization_image, metrics_text, error)
    """
    if file is None:
        return None, None, None, "Please upload an X-ray image"
    
    try:
        # Read file
        if isinstance(file, str):
            # File path
            with open(file, 'rb') as f:
                file_bytes = f.read()
            filename = os.path.basename(file)
        else:
            # File object
            file_bytes = file
            filename = "uploaded_image.jpg"
        
        return predict_xray(file_bytes, filename)
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        return None, None, None, error_msg


# Flask REST API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file
        file_bytes = file.read()
        filename = file.filename
        
        # Validate and preprocess
        is_valid, error = preprocessor.validate_file(file_bytes, filename, MAX_FILE_SIZE_MB)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Predict
        preprocess_start = time.time()
        img_tensor, original_image, metadata = preprocessor.process_file(file_bytes, filename)
        img_tensor = img_tensor.to(device)
        preprocess_time = time.time() - preprocess_start
        
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]
        inference_time = time.time() - inference_start
        
        # Format response
        pred_dict = {label: float(prob) for label, prob in zip(pathology_labels, predictions)}
        sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        
        response = {
            'predictions': [
                {'pathology': label, 'probability': prob}
                for label, prob in sorted_preds
            ],
            'top_prediction': {
                'pathology': sorted_preds[0][0],
                'probability': sorted_preds[0][1]
            },
            'latency_metrics': {
                'preprocessing_ms': round(preprocess_time * 1000, 2),
                'inference_ms': round(inference_time * 1000, 2),
                'total_ms': round((preprocess_time + inference_time) * 1000, 2)
            },
            'model_info': {
                'name': MODEL_NAME,
                'device': str(device),
                'pathologies': pathology_labels
            },
            'image_metadata': metadata
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"API Error: {error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'pathologies_count': len(pathology_labels) if pathology_labels else 0
    }), 200


# Create Gradio interface
def create_gradio_interface():
    """Create Gradio web interface."""
    
    with gr.Blocks(
        title="Chest X-Ray Pathology Predictor",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🏥 Chest X-Ray Pathology Predictor
            
            Upload a chest X-ray image to detect potential pathologies using deep learning.
            Supports **JPG, PNG, and DICOM** formats.
            
            **Features:**
            - Multi-pathology detection using DenseNet-121
            - Grad-CAM visualization showing model attention
            - Support for standard images and DICOM files
            - Automatic DICOM de-identification
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Chest X-Ray",
                    file_types=[".jpg", ".jpeg", ".png", ".dcm", ".dicom"],
                    type="binary"
                )
                predict_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
                
                gr.Markdown("### 📊 Performance Metrics")
                metrics_output = gr.Markdown()
                
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Prediction Results")
                results_output = gr.Dataframe(
                    label="Top Predicted Pathologies",
                    headers=["Pathology", "Probability", "Percentage"],
                    wrap=True
                )
                
                gr.Markdown("### 🔥 Grad-CAM Visualization")
                gr.Markdown("*Left: Original | Right: Heatmap Overlay*")
                viz_output = gr.Image(label="Attention Map", type="pil")
                
                error_output = gr.Textbox(label="Error", visible=True, lines=3)
        
        # Wire up the interface
        predict_btn.click(
            fn=gradio_predict,
            inputs=[file_input],
            outputs=[results_output, viz_output, metrics_output, error_output]
        )
        
        gr.Markdown(
            """
            ---
            **Model:** DenseNet-121 trained on multiple chest X-ray datasets  
            **Technology:** torchxrayvision, Grad-CAM, Flask, Gradio
            
            **REST API:** Access programmatic predictions at `/api/predict`
            """
        )
    
    return demo


if __name__ == '__main__':
    import threading
    
    # Load model at startup
    load_model()
    
    # Create Gradio interface
    gradio_app = create_gradio_interface()
    
    # Function to run Flask in background thread
    def run_flask():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    # Start Flask API server in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("\n" + "="*50)
    print("Starting Flask + Gradio Application")
    print("="*50)
    print(f"Gradio UI:  http://localhost:7860")
    print(f"REST API:   http://localhost:5000/api/predict")
    print(f"Health:     http://localhost:5000/health")
    print("="*50 + "\n")
    
    # Launch Gradio
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


