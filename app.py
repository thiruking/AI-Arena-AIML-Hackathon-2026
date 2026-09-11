#!/usr/bin/env python3
"""
CropGuard Web Server & Localhost Runner
Provides REST API & Interactive UI for CropGuard 6-Class Leaf Disease Classification.
"""
import os
import io
import sys
import base64
import argparse
import warnings
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

# Suppress scikit-learn version mismatch warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)

# Import feature extraction logic from the project
from features import extract_features, load_and_extract

# Initialize Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "templates"))

# Path to model bundle
MODEL_PATH = os.path.join(BASE_DIR, "cropguard_model.joblib")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

print(f"[*] Loading CropGuard model bundle from {MODEL_PATH}...")
try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    classes = bundle["classes"]
    model_name = bundle.get("model_name", "VotingClassifier Ensemble")
    val_accuracy = bundle.get("val_accuracy", 0.9333)
    print(f"[+] Successfully loaded '{model_name}' (Held-out Val Accuracy: {val_accuracy * 100:.2f}%)")
    print(f"[+] Classes: {classes}")
except Exception as e:
    print(f"[-] Error loading model bundle: {e}")
    sys.exit(1)

# Disease pathology metadata & agronomic advisory
DISEASE_METADATA = {
    "blight": {
        "name": "Foliar Blight",
        "pathogen": "Phytophthora / Alternaria complex",
        "severity": "High / Critical",
        "severity_level": "danger",
        "symptoms": "Rapidly spreading water-soaked necrotic lesions with dark concentric rings and chlorotic borders.",
        "treatment": "Apply copper oxychloride (2.5g/L) or Azoxystrobin + Difenoconazole systemic fungicide. Discontinue overhead watering immediately.",
        "prevention": "Sterilize pruning tools, ensure 18+ inch plant spacing for canopy ventilation, and practice a 3-year crop rotation."
    },
    "healthy": {
        "name": "Healthy Plant Leaf",
        "pathogen": "None (Optimum Foliage)",
        "severity": "Optimal",
        "severity_level": "healthy",
        "symptoms": "Vigorous photosynthetic lamina, uniform chlorophyll distribution, intact leaf margins, and clear vein architecture.",
        "treatment": "No chemical treatment needed. Continue standard balanced N-P-K nutrient feeding and automated soil moisture monitoring.",
        "prevention": "Maintain preventative IPM (Integrated Pest Management) and regular scouting schedules."
    },
    "leaf_spot": {
        "name": "Cercospora / Septoria Leaf Spot",
        "pathogen": "Cercospora / Septoria fungi",
        "severity": "Moderate",
        "severity_level": "warning",
        "symptoms": "Circular to angular lesions with tan/grey centers and prominent dark margins on mature leaves.",
        "treatment": "Apply Chlorothalonil or Mancozeb protective spray. For organic cultivation, apply Bacillus subtilis bio-fungicide weekly.",
        "prevention": "Collect and destroy fallen leaf litter to eliminate fungal inoculum overwintering in topsoil."
    },
    "mildew": {
        "name": "Powdery / Downy Mildew",
        "pathogen": "Oidium / Peronospora fungal hyphae",
        "severity": "Moderate to High",
        "severity_level": "warning",
        "symptoms": "White-to-grey talcum-like fungal coating across the adaxial leaf surface, accompanied by leaf curling and stunting.",
        "treatment": "Apply Potassium Bicarbonate foliar spray (3g/L) or cold-pressed Neem Oil emulsion (5ml/L). In severe cases, apply Myclobutanil.",
        "prevention": "Reduce microclimate humidity, improve sunlight exposure, and switch to morning subsurface drip irrigation."
    },
    "mosaic": {
        "name": "Viral Mosaic Complex",
        "pathogen": "Plant Potyvirus / Tobamovirus",
        "severity": "High (Systemic)",
        "severity_level": "danger",
        "symptoms": "Irregular light and dark green mosaic variegation, puckering, vein-clearing, and asymmetric lamina growth.",
        "treatment": "Viral infections are incurable once systemic. Rogue and incinerate infected plants immediately to protect surrounding crops.",
        "prevention": "Aggressively manage sap-sucking insect vectors (aphids, thrips, whiteflies) with insecticidal soap or reflective mulches."
    },
    "rust": {
        "name": "Rust Fungus",
        "pathogen": "Puccinia / Uromyces rust fungi",
        "severity": "Severe",
        "severity_level": "danger",
        "symptoms": "Raised reddish-brown to orange uredinial pustules rupturing the epidermis, producing powdery spores on leaf undersides.",
        "treatment": "Apply Triazole fungicides (Propiconazole / Tebuconazole) or Wettable Micronized Sulfur. Treat adjacent buffer rows.",
        "prevention": "Select rust-resistant cultivars, eradicate alternative weed hosts, and prevent moisture buildup on leaves."
    }
}


def process_bgr_image(bgr_img):
    """Run feature extraction, scaling, and ensemble inference on a BGR image."""
    feat = extract_features(bgr_img).reshape(1, -1)
    feat_s = scaler.transform(feat)
    
    proba = model.predict_proba(feat_s)[0]
    idx = int(np.argmax(proba))
    
    pred_class = model.classes_[idx] if hasattr(model, "classes_") else classes[idx]
    confidence = float(proba[idx])
    
    prob_dict = {}
    sorted_probs = []
    for c_idx, c_name in enumerate(classes):
        p = float(proba[c_idx])
        prob_dict[c_name] = round(p, 4)
        sorted_probs.append({
            "class_name": c_name,
            "display_name": DISEASE_METADATA.get(c_name, {}).get("name", c_name.title()),
            "probability": round(p, 4),
            "percentage": round(p * 100, 2)
        })
    sorted_probs.sort(key=lambda x: x["probability"], reverse=True)
    
    hsv = cv2.cvtColor(cv2.resize(bgr_img, (128, 128)), cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cv2.resize(bgr_img, (128, 128)), cv2.COLOR_BGR2LAB)
    
    feature_insights = {
        "total_dimensions": int(feat.shape[1]),
        "mean_hue": round(float(hsv[:, :, 0].mean()), 2),
        "mean_saturation": round(float(hsv[:, :, 1].mean()), 2),
        "mean_value": round(float(hsv[:, :, 2].mean()), 2),
        "lab_a_channel_mean": round(float(lab[:, :, 1].mean()), 2),
        "lab_b_channel_mean": round(float(lab[:, :, 2].mean()), 2),
        "feature_vector_norm": round(float(np.linalg.norm(feat)), 3)
    }
    
    meta = DISEASE_METADATA.get(pred_class, {
        "name": pred_class.replace("_", " ").title(),
        "pathogen": "Unknown",
        "severity": "Unknown",
        "severity_level": "warning",
        "symptoms": "N/A",
        "treatment": "N/A",
        "prevention": "N/A"
    })
    
    return {
        "predicted_class": pred_class,
        "disease_name": meta["name"],
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "pathogen": meta["pathogen"],
        "severity": meta["severity"],
        "severity_level": meta["severity_level"],
        "is_healthy": (pred_class == "healthy"),
        "symptoms": meta["symptoms"],
        "treatment": meta["treatment"],
        "prevention": meta["prevention"],
        "probabilities": prob_dict,
        "sorted_probabilities": sorted_probs,
        "feature_insights": feature_insights
    }


# =====================================================================
# API ROUTES
# =====================================================================

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html",
                           model_name=model_name,
                           val_accuracy=round(val_accuracy * 100, 2),
                           classes=classes)


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check and model metadata."""
    return jsonify({
        "status": "healthy",
        "service": "CropGuard AI Disease Diagnostic System",
        "model_name": model_name,
        "val_accuracy": round(val_accuracy * 100, 2),
        "classes": list(classes),
        "num_classes": len(classes),
        "feature_dimensions": 1688,
        "python_version": sys.version.split()[0]
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Return list of available sample images for each class."""
    sample_catalog = []
    if os.path.isdir(SAMPLES_DIR):
        for c in sorted(os.listdir(SAMPLES_DIR)):
            c_dir = os.path.join(SAMPLES_DIR, c)
            if os.path.isdir(c_dir):
                files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    sample_catalog.append({
                        "class_name": c,
                        "display_name": DISEASE_METADATA.get(c, {}).get("name", c.title()),
                        "filename": files[0],
                        "severity_level": DISEASE_METADATA.get(c, {}).get("severity_level", "healthy")
                    })
    return jsonify({"samples": sample_catalog})


@app.route("/api/sample/<class_name>", methods=["GET"])
def api_sample_by_class(class_name):
    """Retrieve base64 image and info for a sample class."""
    class_name = class_name.lower().strip()
    c_dir = os.path.join(SAMPLES_DIR, class_name)
    if not os.path.isdir(c_dir):
        return jsonify({"error": f"Sample class '{class_name}' not found."}), 404
    
    files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return jsonify({"error": f"No image found for class '{class_name}'."}), 404
    
    file_path = os.path.join(c_dir, files[0])
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    b64_str = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
    
    return jsonify({
        "class_name": class_name,
        "filename": files[0],
        "image_data": b64_str
    })


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    """Run real-time diagnosis on an uploaded image or sample class."""
    try:
        bgr_img = None
        img_filename = "custom_leaf_scan.png"
        
        # 1. Check if multipart file upload
        if "image" in request.files:
            file = request.files["image"]
            if file.filename:
                img_filename = file.filename
                in_memory = file.read()
                nparr = np.frombuffer(in_memory, np.uint8)
                bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
        # 2. Check JSON payload (base64 or sample_class)
        elif request.is_json:
            data = request.get_json()
            if "sample_class" in data:
                sample_cls = data["sample_class"].lower().strip()
                c_dir = os.path.join(SAMPLES_DIR, sample_cls)
                if os.path.isdir(c_dir):
                    files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        file_path = os.path.join(c_dir, files[0])
                        img_filename = files[0]
                        bgr_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            elif "image_data" in data:
                b64_data = data["image_data"]
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if bgr_img is None:
            return jsonify({"error": "No valid image provided. Upload a JPG/PNG file or select a sample."}), 400

        result = process_bgr_image(bgr_img)
        result["filename"] = img_filename
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Diagnosis failed: {str(e)}"}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Run batch classification on multiple uploaded files and return CSV rows."""
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided in request."}), 400
        
        files = request.files.getlist("files")
        results = []
        
        for file in files:
            if not file.filename:
                continue
            sample_id = os.path.splitext(file.filename)[0]
            in_memory = file.read()
            nparr = np.frombuffer(in_memory, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                continue
            
            res = process_bgr_image(bgr_img)
            results.append({
                "sample_id": sample_id,
                "predicted_class": res["predicted_class"],
                "confidence": res["confidence"]
            })
            
        csv_lines = ["sample_id,predicted_class,confidence"]
        for r in results:
            csv_lines.append(f"{r['sample_id']},{r['predicted_class']},{r['confidence']}")
        csv_text = "\n".join(csv_lines)
        
        return jsonify({
            "count": len(results),
            "predictions": results,
            "csv_data": csv_text
        })
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


def main():
    parser = argparse.ArgumentParser(description="Run CropGuard Localhost Web Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    print("================================================================")
    print("      CropGuard(TM) - AI Plant Health & Disease Diagnostic")
    print(f"      Localhost Server starting on http://{args.host}:{args.port}")
    print("================================================================")
    print(f"[*] Open http://localhost:{args.port} in your browser.")
    print("================================================================")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
