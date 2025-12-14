Test A — Build a Flask UI service around torchxrayvision
Brief
Create a minimal clinical-style web app that serves pretrained models from [mlmed/torchxrayvision] and lets users upload chest X-ray images to obtain predicted pathologies and a visual explanation map. Provide both:
a browser UI (templated pages), and

a JSON REST API for programmatic access.

Core Requirements
Model

Load a pretrained model (e.g., densenet121-res224-all) from torchxrayvision.

Warm-load the model at app start and reuse the instance per request.

Input

Accept JPG/PNG and DICOM. For DICOM, de-identify (strip metadata) before storing/processing.

Validate file size (≤ 10 MB) and type; reject anything else with a clear error.

Preprocessing

Convert to grayscale if needed, normalize per the model’s expected transform, and resize appropriately.

Prediction

Return class probabilities for the model’s known labels (e.g., pathologies).

Include latency metrics in the response (preprocess time, inference time).

Explainability

Provide a Grad-CAM-style heatmap overlay (or Captum integrated gradients) for at least one predicted class.

UI

Simple upload form, results table for probabilities, and a side-by-side image + heatmap viewer.
