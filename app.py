"""
CornOrb – Keratoconus Diagnostic System
========================================
Flask web application that provides automated keratoconus diagnosis
using a pretrained Multi-Map ViT-B/16 model.

Supports three input workflows:
  1. Upload four individual corneal map images (Axial, Anterior, Posterior, Pachymetry)
  2. Upload a single composite image containing four maps + clinical tables
  3. Upload a PDF containing four maps + clinical tables

The application will crop/extract maps, display them, optionally extract
clinical features from tables, and run inference through the deep learning model.
"""

import os
import io
import re
import uuid
import json
import base64
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import pytesseract

import torch
import torch.nn as nn
import timm
from torchvision import transforms

from flask import Flask, render_template, request, jsonify

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Configuration ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = BASE_DIR / "MultiMap_ViT_B16_best.pth"

IMG_SIZE = 224
IN_CHANNELS = 12  # 4 maps × 3 RGB channels
NUM_CLASSES = 2
CLASS_LABELS = {0: "Normal", 1: "Keratoconus"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image Preprocessing ────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model Definition ───────────────────────────────────────────────────────────
class MultiMapViT(nn.Module):
    """
    ViT-B/16 modified for 12-channel multi-map keratoconus classification.
    
    Architecture:
    ─────────────
    Input [B, 12, 224, 224]
        ↓ Patch Embedding: Conv2d(12, 768, 16×16) → 196 patches
        ↓ + CLS token + positional encoding
        ↓ Transformer Blocks 0–7 (frozen)
        ↓ Transformer Blocks 8–11 (trainable)
        ↓ CLS token output → [B, 768]
        ↓ LayerNorm
        ↓ Linear(768→256) + GELU + Dropout
        ↓ Linear(256→2)
    Output: logits [B, 2]
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 in_channels: int = IN_CHANNELS, dropout: float = 0.3):
        super().__init__()

        # Load pretrained ViT-B/16 via timm
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,
            drop_rate=dropout,
            attn_drop_rate=0.1
        )

        # ── Replace Patch Embedding Projection ──────────────────────────────
        old_proj = self.backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None)
        )
        self.backbone.patch_embed.proj = new_proj

        # ── Freeze first 8 transformer blocks ───────────────────────────────
        total_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            if i < total_blocks - 4:
                for param in block.parameters():
                    param.requires_grad = False

        # Always keep patch embedding trainable
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = True

        # ── Custom classification head ──────────────────────────────────────
        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model():
    """Load the pretrained ViT model from checkpoint."""
    model = MultiMapViT().to(DEVICE)
    if WEIGHTS_PATH.exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[OK] Model loaded from {WEIGHTS_PATH}")
        print(f"   Best validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        print(f"[WARN] No weights found at {WEIGHTS_PATH}. Model will use random weights.")
    model.eval()
    return model


# ── Helper Functions ────────────────────────────────────────────────────────────
def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def preprocess_map_image(img: Image.Image) -> torch.Tensor:
    """Apply inference transforms to a single map image."""
    img_rgb = img.convert("RGB")
    return inference_transform(img_rgb)


def run_inference(model, map_images: list) -> dict:
    """
    Run inference on 4 map images.
    
    Args:
        model: The loaded ViT model
        map_images: List of 4 PIL Images [Axial, Anterior, Posterior, Pachymetry]
    
    Returns:
        Dictionary with prediction results
    """
    if len(map_images) != 4:
        raise ValueError(f"Expected 4 map images, got {len(map_images)}")

    # Preprocess each map and stack into 12-channel tensor
    tensors = [preprocess_map_image(img) for img in map_images]
    # Each tensor is [3, 224, 224], concatenate along channel dim → [12, 224, 224]
    combined = torch.cat(tensors, dim=0).unsqueeze(0)  # [1, 12, 224, 224]
    combined = combined.to(DEVICE)

    with torch.no_grad():
        logits = model(combined)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return {
        "prediction": CLASS_LABELS[pred_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "Normal": round(probs[0, 0].item() * 100, 2),
            "Keratoconus": round(probs[0, 1].item() * 100, 2),
        }
    }


def crop_maps_from_composite(composite_img: Image.Image) -> list:
    """
    Crop 4 individual maps from an Orbscan-style composite image.
    
    Layout (based on actual Orbscan output):
    ─────────────────────────────────────────────────────────
    | Patient Data    | Axial/Sagittal    | Elevation Front  |
    | & Tables        | Curvature (Map1)  | (Map2)           |
    | (~35% width)    |                   |                  |
    |─────────────────|───────────────────|──────────────────|
    | More Tables     | Corneal Thickness | Elevation Back   |
    | & Pachy Data    | (Map3)            | (Map4)           |
    ─────────────────────────────────────────────────────────
    """
    w, h = composite_img.size
    
    # Left column (patient data/tables) takes ~36% of width
    # The 4 maps are in the right ~64%, arranged as 2 columns × 2 rows
    data_col_end = int(w * 0.36)       # End of data column (increased to avoid text in maps)
    mid_col = int(w * 0.68)            # Border between middle and right map columns
    mid_row = int(h * 0.50)            # Border between top and bottom map rows
    
    # Small padding to trim titles/borders around each map
    pad_top = int(h * 0.04)
    pad_bottom = int(h * 0.04)
    pad_left = int(w * 0.02)
    pad_right = int(w * 0.01)
    
    # Crop the four maps (Order must be: Axial, Anterior Elevation, Posterior Elevation, Pachymetry)
    maps = [
        # Map 1: Axial / Sagittal Curvature (top-middle)
        composite_img.crop((
            data_col_end + pad_left,
            pad_top,
            mid_col - pad_right,
            mid_row - pad_bottom
        )),
        # Map 2: Elevation Front (top-right)
        composite_img.crop((
            mid_col + pad_left,
            pad_top,
            w - pad_right,
            mid_row - pad_bottom
        )),
        # Map 3: Elevation Back (bottom-right)
        composite_img.crop((
            mid_col + pad_left,
            mid_row + pad_top,
            w - pad_right,
            h - pad_bottom
        )),
        # Map 4: Corneal Thickness / Pachymetry (bottom-middle)
        composite_img.crop((
            data_col_end + pad_left,
            mid_row + pad_top,
            mid_col - pad_right,
            h - pad_bottom
        )),
    ]
    
    return maps


def extract_clinical_data(composite_img: Image.Image) -> list:
    """
    Extract clinical data from the left panel of the Orbscan composite image
    using OCR. Returns raw extracted values without risk assessment.
    
    Returns a list of dicts: [{name, value, unit}]
    """
    from PIL import ImageFilter, ImageEnhance
    
    w, h = composite_img.size
    # Crop left 33% — the data tables column
    data_col_end = int(w * 0.33)
    data_region = composite_img.crop((0, 0, data_col_end, h))
    
    # ── Image preprocessing for OCR ──
    # 1) Convert to grayscale
    gray = data_region.convert('L')
    # 2) Increase contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    # 3) Sharpen
    gray = gray.filter(ImageFilter.SHARPEN)
    # 4) Scale up 3x for better character recognition
    gray = gray.resize((gray.width * 3, gray.height * 3), Image.LANCZOS)
    # 5) Threshold: keep dark text, make background white
    gray = gray.point(lambda p: 0 if p < 140 else 255)
    
    try:
        raw_text = pytesseract.image_to_string(gray, config='--psm 6 --oem 3')
    except Exception as e:
        print(f"[WARN] OCR failed: {e}")
        return []
    
    print(f"[OCR] Raw text:\n{raw_text}")
    
    # ── Helper to find numeric values ──
    def find_val(patterns, text):
        """Try multiple regex patterns, return first match as float."""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                try:
                    val_str = m.group(1).strip().replace(',', '.').replace(' ', '')
                    return float(val_str)
                except (ValueError, IndexError):
                    pass
        return None

    clinical_data = []
    
    # Helper to add a found value
    def add(name, val, unit, risk_level="normal", risk_label="Normal"):
        if val is not None:
            clinical_data.append({
                "name": name, 
                "value": val, 
                "unit": unit,
                "risk": risk_level,
                "risk_label": risk_label
            })
        else:
            clinical_data.append({
                "name": name, 
                "value": "N/A", 
                "unit": "",
                "risk": "normal",
                "risk_label": "—"
            })
    
    # ──────────────────── Cornea Front ────────────────────
    front_section = raw_text.split('Cornea Back')[0] if 'Cornea Back' in raw_text else raw_text.split('Back')[0] if 'Back' in raw_text else raw_text
    back_section = raw_text.split('Cornea Back')[1] if 'Cornea Back' in raw_text else (raw_text.split('Back')[1] if 'Back' in raw_text else '')
    pachy_section = raw_text[raw_text.lower().find('pach'):] if 'pach' in raw_text.lower() else raw_text
    
    # K1 Front
    v = find_val([
        r'K1[^\[0-9]*\[?\s*([345]\d[.,]\d+)',
        r'K[i1][^\[0-9]*\[?\s*([345]\d[.,]\d+)',
        r'Ka[^\[0-9]*\[?\s*([345]\d[.,]\d+)',
    ], front_section)
    add("K1 (Front)", v, "D", "normal", "—")
    
    # K2 Front
    v = find_val([
        r'K2[^\[0-9]*\[?\s*([345]\d[.,]\d+)',
        r'K[z2][^\[0-9]*\[?\s*([345]\d[.,]\d+)',
    ], front_section)
    add("K2 (Front)", v, "D", "normal", "—")
    
    # Km Front
    v = find_val([
        r'Km[^\[0-9]*\[?\s*([345]\d[.,]\d+)',
    ], front_section)
    add("Km (Front)", v, "D", "normal", "—")
    
    # Astigmatism Front
    v = find_val([
        r'Asti[gq]?[^\[0-9]*\[?\s*([+-]?\d+[.,]?\d*)',
    ], front_section)
    add("Astigmatism (Front)", v, "D", "normal", "—")
    
    # Q-value Front
    v = find_val([
        r'Q[-.]val[^\[0-9]*\[?\s*([+-]?\s*\d+[.,]\d+)',
    ], front_section)
    add("Q-value (Front)", v, "", "normal", "—")
    
    # Rmin Front  
    v = find_val([
        r'R\s*min[^\[0-9]*\[?\s*(\d+[.,]\d+)',
        r'Rimin[^\[0-9]*\[?\s*(\d+[.,]\d+)',
    ], front_section)
    add("Rmin (Front)", v, "mm", "normal", "—")
    
    # ──────────────────── Cornea Back ────────────────────
    # K1 Back
    v = find_val([
        r'K1[^\[0-9]*\[?\s*(-?\d+[.,]\d+)',
        r'K[i1][^\[0-9]*\[?\s*(-?\d+[.,]\d+)',
    ], back_section)
    add("K1 (Back)", v, "D", "normal", "—")
    
    # K2 Back
    v = find_val([
        r'K2[^\[0-9]*\[?\s*(-?\d+[.,]\d+)',
        r'K[z2][^\[0-9]*\[?\s*(-?\d+[.,]\d+)',
    ], back_section)
    add("K2 (Back)", v, "D", "normal", "—")
    
    # Km Back
    v = find_val([
        r'Km[^\[0-9]*\[?\s*(-?\d+[.,]\d+)',
    ], back_section)
    add("Km (Back)", v, "D", "normal", "—")
    
    # Astigmatism Back
    v = find_val([
        r'Asti[gq]?[^\[0-9]*\[?\s*([+-]?\d+[.,]?\d*)',
    ], back_section)
    add("Astigmatism (Back)", v, "D", "normal", "—")
    
    # ──────────────────── Pachymetry & Rules ────────────────────
    # User Rules: Thickness > 500 (Normal), 480-500 (Suspicious), < 470 (Abnormal)
    # Pupil Center
    v = find_val([
        r'Pupil\s*Center[^\[0-9]*\[?\s*(\d{3,})',
    ], pachy_section)
    if v is not None:
        risk = "normal" if v > 500 else ("danger" if v < 470 else "warning")
        label = "Normal" if risk == "normal" else ("Abnormal" if risk == "danger" else "Suspicious")
        add("Pachy (Pupil Center)", v, "µm", risk, label)
    
    # Pachy Apex
    v = find_val([
        r'Pachy\s*[Aa]pex[^\[0-9]*\[?\s*(\d{3,})',
    ], pachy_section)
    if v is not None:
        risk = "normal" if v > 500 else ("danger" if v < 470 else "warning")
        label = "Normal" if risk == "normal" else ("Abnormal" if risk == "danger" else "Suspicious")
        add("Pachy (Apex)", v, "µm", risk, label)
    
    # Thinnest Location
    thinnest_v = find_val([
        r'Thinnest\s*Locat?[^\[0-9]*\[?\s*(\d{3,})',
        r'Thinnest[^\[0-9]*\[?\s*(\d{3,})',
    ], pachy_section)
    if thinnest_v is not None:
        risk = "normal" if thinnest_v > 500 else ("danger" if thinnest_v < 470 else "warning")
        label = "Normal" if risk == "normal" else ("Abnormal (Keratoconus Sign)" if risk == "danger" else "Suspicious")
        add("Thinnest Pachymetry", thinnest_v, "µm", risk, label)
    
    # ──────────────────── Other Key Values ────────────────────
    # User Rules: Sagittal Power < 48 (Normal), 48-49 (Suspicious), > 49 (Abnormal)
    # K Max (Front)
    kmax_v = find_val([
        r'K\s*Max[^\[0-9]*\[?\s*([3456]\d[.,]\d+)',
        r'KMax[^\[0-9]*\[?\s*([3456]\d[.,]\d+)',
    ], raw_text)
    if kmax_v is not None:
        risk = "normal" if kmax_v < 48 else ("danger" if kmax_v >= 49 else "warning")
        label = "Normal" if risk == "normal" else ("Abnormal (Keratoconus Sign)" if risk == "danger" else "Suspicious")
        add("K Max (Front)", kmax_v, "D", risk, label)
    
    # Cornea Volume
    v = find_val([
        r'Co[rm]n?ea\s*Volume[^\[0-9]*\[?\s*(\d+[.,]\d+)',
    ], raw_text)
    add("Cornea Volume", v, "mm³", "normal", "—")
    
    # A.C. Depth
    v = find_val([
        r'A\.?C\.?\s*Depth[^\[0-9]*\[?\s*(\d+[.,]?\d*)',
    ], raw_text)
    if v and v > 100: v = v / 100.0
    elif v and v > 10: v = v / 10.0
    add("A.C. Depth", v, "mm", "normal", "—")
    
    # Pupil Diameter
    v = find_val([
        r'Pupil\s*Di[ae][^\[0-9]*\[?\s*(\d+[.,]?\d*)',
    ], raw_text)
    if v and v > 100: v = v / 100.0
    elif v and v > 10: v = v / 10.0
    add("Pupil Diameter", v, "mm", "normal", "—")
    
    # Evaluate Clinical Overall Rule-Based Diagnosis
    clinical_diagnosis = "Normal"
    clinical_risk = "normal"
    has_danger = False
    has_warning = False
    
    if kmax_v is not None:
        if kmax_v >= 49: has_danger = True
        elif kmax_v >= 48: has_warning = True
        
    if thinnest_v is not None:
        if thinnest_v < 470: has_danger = True
        elif thinnest_v <= 500: has_warning = True
        
    if has_danger:
        clinical_diagnosis = "Keratoconus (Clinical Rules)"
        clinical_risk = "danger"
    elif has_warning:
        clinical_diagnosis = "Suspicious (Clinical Rules)"
        clinical_risk = "warning"
    else:
        clinical_diagnosis = "Normal (Clinical Rules)"
        clinical_risk = "normal"
        
    return {
        "values": clinical_data,
        "overall_diagnosis": clinical_diagnosis,
        "overall_risk": clinical_risk
    }

def extract_maps_from_pdf(pdf_path: str) -> list:
    """
    Extract map images from a PDF file.
    Renders each page at high resolution and attempts to crop maps.
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    all_maps = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 2x resolution for better quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Try to crop maps from this page
        maps = crop_maps_from_composite(img)
        all_maps.extend(maps)
        
        if len(all_maps) >= 4:
            break
    
    doc.close()
    return all_maps[:4]


# ── Flask Application ───────────────────────────────────────────────────────────
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Load model at startup
model = load_model()


@app.route('/')
def index():
    """Serve the home page."""
    return render_template('index.html')


@app.route('/diagnose')
def diagnose():
    """Serve the diagnostic page."""
    return render_template('diagnose.html')


@app.route('/about')
def about():
    """Serve the about page."""
    return render_template('about.html')


@app.route('/api/diagnose/individual', methods=['POST'])
def diagnose_individual():
    """
    Endpoint for Mode 1: Four individual map images.
    Expects files named: axial, anterior, posterior, pachymetry
    """
    try:
        map_names = ['axial', 'anterior', 'posterior', 'pachymetry']
        map_images = []
        map_previews = []

        for name in map_names:
            if name not in request.files:
                return jsonify({"error": f"Missing map: {name}"}), 400
            
            file = request.files[name]
            if file.filename == '':
                return jsonify({"error": f"No file selected for {name}"}), 400
            
            img = Image.open(file.stream)
            map_images.append(img)
            map_previews.append({
                "name": name.capitalize(),
                "data": image_to_base64(img.copy())
            })

        # Run inference
        result = run_inference(model, map_images)
        result["maps"] = map_previews
        result["mode"] = "individual"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/diagnose/composite', methods=['POST'])
def diagnose_composite():
    """
    Endpoint for Mode 2: Single composite image with 4 maps + tables.
    Expects a file named: composite
    """
    try:
        if 'composite' not in request.files:
            return jsonify({"error": "No composite image uploaded"}), 400
        
        file = request.files['composite']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        composite_img = Image.open(file.stream)
        
        # Crop individual maps
        map_images = crop_maps_from_composite(composite_img)
        map_names = ['Axial / Sagittal Curvature', 'Elevation (Front)', 'Elevation (Back)', 'Corneal Thickness']
        
        map_previews = []
        for i, (img, name) in enumerate(zip(map_images, map_names)):
            map_previews.append({
                "name": name,
                "data": image_to_base64(img.copy())
            })

        # Full composite preview
        composite_preview = image_to_base64(composite_img.copy())

        # Extract clinical data via OCR
        # Save a copy for debugging
        debug_path = UPLOAD_DIR / "last_composite.png"
        composite_img.save(str(debug_path))
        print(f"[DEBUG] Saved composite to {debug_path}")
        clinical_data = extract_clinical_data(composite_img)
        print(f"[DEBUG] Extracted {len(clinical_data)} clinical values: {clinical_data}")

        # Run inference
        result = run_inference(model, map_images)
        result["maps"] = map_previews
        result["composite_preview"] = composite_preview
        result["clinical_data"] = clinical_data
        result["mode"] = "composite"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/diagnose/pdf', methods=['POST'])
def diagnose_pdf():
    """
    Endpoint for Mode 3: PDF file with maps + tables.
    Expects a file named: pdf_file
    """
    try:
        if 'pdf_file' not in request.files:
            return jsonify({"error": "No PDF file uploaded"}), 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save PDF temporarily
        pdf_filename = f"{uuid.uuid4().hex}.pdf"
        pdf_path = UPLOAD_DIR / pdf_filename
        file.save(str(pdf_path))
        
        try:
            # Extract maps from PDF
            map_images = extract_maps_from_pdf(str(pdf_path))
            
            if len(map_images) < 4:
                return jsonify({
                    "error": f"Could only extract {len(map_images)} maps from PDF. Need 4."
                }), 400

            map_names = ['Axial / Sagittal Curvature', 'Elevation (Front)', 'Elevation (Back)', 'Corneal Thickness']
            map_previews = []
            for img, name in zip(map_images, map_names):
                map_previews.append({
                    "name": name,
                    "data": image_to_base64(img.copy())
                })

            # Run inference
            result = run_inference(model, map_images)
            result["maps"] = map_previews
            result["mode"] = "pdf"

            return jsonify(result)
        
        finally:
            # Clean up temporary file
            if pdf_path.exists():
                pdf_path.unlink()

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=True is disabled for deployment as it causes signal handler errors on many platforms
    app.run(debug=False, host='0.0.0.0', port=port)
