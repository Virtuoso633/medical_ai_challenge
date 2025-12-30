
# 🫁 Enhanced X-Ray Pathology Predictor (Hybrid Ensemble)

This repository contains a professional-grade chest X-ray screening tool that combines local deep learning ensembles with Google's **CXR Foundation** model to achieve high-accuracy, zero-shot pathology prediction.


## Gradio Dashboard
<img width="2560" height="1351" alt="Screenshot 2025-12-30 at 6 31 58 AM" src="https://github.com/user-attachments/assets/d13e8fdd-5ac9-491b-816a-5c92839e7da7" />

## 📋 What it does

The system predicts **18 common pathologies** (Atelectasis, Pneumonia, Cardiomegaly, etc.) from standard chest X-ray images (JPG, PNG, or DICOM).

- **Hybrid Detection**: Merges local predictions with foundation model intelligence.
- **Visual Explanations**: Generates Attention Heatmaps (Grad-CAM) for every finding.
- **Clinical Thresholds**: Uses per-pathology optimized thresholds to minimize false positives.
- **Performance**: Real-time analysis (< 2s) using hardware-accelerated inference.

## ⚙️ How it works (Architecture)

The system employs a **Hybrid Cloud-Local Ensemble** strategy:

1.  **Local Engine (PyTorch)**: A weighted ensemble of 5 DenseNet-121 models, each pre-trained on diverse datasets (NIH, CheXpert, MIMIC-CXR, and PadChest).
2.  **Cloud Engine (TensorFlow)**: A remote connection to Google's **CXR Foundation** model running on a GPU-enabled Google Colab instance.
3.  **Combination Layer**: A weighted fusion logic (70% Local / 30% Cloud) that balances widespread dataset knowledge with foundation-scale zero-shot reasoning.
4.  **Hardware Optimization**:
    - **Local**: Automatically utilizes **Apple Metal (MPS)** on Mac or **CUDA** on Linux/Windows.
    - **Cloud**: Uses **NVIDIA T4** GPUs via the Gradio API.

## 🧠 Why this approach?

- **Superior Accuracy**: Foundation models (like Google's ELIXR) provide better generalization for rare findings than standalone supervised models.
- **Scalability**: By running the heavy foundation model in the cloud, users can run the app on standard laptops without high-end GPUs.
- **Radiologist-Centric**: "Model Agreement" scores show how much the internal models agree, highlighting high-uncertainty cases for human review.

## 🚀 Setup Guide (New Developers)

### 1. Local Prerequisites

Ensure you have Python 3.10+ installed. Then install the core dependencies:

```bash
pip install torch torchxrayvision gradio gradio_client pillow pandas numpy
```

### 2. Cloud Service Setup (Google Colab)

The CXR Foundation model requires TensorFlow and a GPU.

1.  Locate `enhanced_ensemble/cxr_foundation_colab.ipynb`.
2.  Upload it to [Google Colab](https://colab.research.google.com/).
3.  Change Runtime to **GPU** (T4).
4.  **Run All Cells** (You will need a HuggingFace token and access to `google/cxr-foundation`).
5.  Copy the public Gradio URL (e.g., `https://xxxx.gradio.live`).

### 3. Launching the Local App

Run the following command from the project root:

```bash
python enhanced_ensemble/app_enhanced.py
```

1.  Open the local URL in your browser.
2.  Paste your **Colab URL** into the "Cloud Integration" field.
3.  Upload an X-ray and click **Analyze Image**.

## 📂 Project Structure

- `app_enhanced.py`: Main Gradio interface and orchestration.
- `ensemble_predictor.py`: Local 5-model ensemble and device-agnostic logic (MPS/CUDA/CPU).
- `cxr_foundation_predictor.py`: Python client for the Colab API.
- `cxr_prompts.py`: Natural language definitions for 18 zero-shot pathologies.
- `threshold_optimizer.py`: Clinical balancing logic for detection status.

---

_Created as part of the X-Ray Pathology Prediction System._
