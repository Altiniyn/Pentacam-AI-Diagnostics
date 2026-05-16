# CornOrb: AI-Powered Keratoconus Diagnostic Portal 👁️🤖

An advanced, AI-driven clinical web application for the early detection and diagnosis of Keratoconus using **Pentacam Corneal Topography** composite maps. 

The system leverages a **Vision Transformer (ViT)** for image-based deep learning inference and integrates a **Computer Vision OCR Pipeline** to extract precise clinical parameters (K Max, Pachymetry, Elevation maps) for rule-based medical evaluation.

---

## 🌟 Key Features

- **Deep Learning Inference:** Utilizes a custom-trained `MultiMap_ViT_B16` PyTorch model to classify cases as Normal or Keratoconus with confidence percentages.
- **Robust Clinical OCR Engine:** Automatically extracts essential patient parameters (K1, K2, Km, Astigmatism, Pachymetry, AC Depth, Max Elevations) from blurry or varied clinical reports using Tesseract and adaptive image preprocessing.
- **Rule-Based Clinical Assessment:** Evaluates extracted metrics against established medical thresholds (e.g., K Max > 49D, Thinnest Pachymetry < 470µm) to provide an independent clinical risk evaluation.
- **Multi-Modal Uploads:** Supports drag-and-drop processing for:
  - 4 Individual Maps (Axial, Front Elevation, Back Elevation, Thickness)
  - Single Composite Pentacam Images
  - PDF Export Reports
- **Dynamic UI/UX:** A responsive, dark-themed medical dashboard designed for ophthalmologists and clinicians.

## 🛠️ Technology Stack

- **Backend:** Python, Flask, PyTorch, PyMuPDF
- **Computer Vision:** OpenCV, PIL, PyTesseract (Tesseract-OCR)
- **Frontend:** Vanilla JS, CSS3, HTML5
- **Architecture:** Hybrid AI + Rule-Based Decision System

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and added to your system path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CornOrb-Keratoconus-AI.git
   cd CornOrb-Keratoconus-AI
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your pre-trained model weights (`MultiMap_ViT_B16_best.pth`) in the root directory. *(Note: Weights are not included in the repo due to size limits).*

### Running the App
Start the Flask local development server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧠 Medical Rules Defined

The clinical rule-based engine evaluates cases independent of the AI model based on:
- **K Max (Front):** `< 48` (Normal) | `48-49` (Suspicious) | `> 49` (Keratoconus Sign)
- **Thinnest Pachymetry:** `> 500` (Normal) | `470-500` (Suspicious) | `< 470` (Keratoconus Sign)
- **Max Elevation (Front):** `> 15` (Abnormal)
- **Max Elevation (Back):** `> 20` (Abnormal)

## 📄 License
This project is for educational and research purposes.

---
*Developed by Abdo & Team*
"# Pentacam-AI-Diagnostics" 
