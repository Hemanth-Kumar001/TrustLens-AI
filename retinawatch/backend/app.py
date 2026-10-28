from pathlib import Path
import json
import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from model import load_model, predict_with_explanation

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": os.environ.get("RETINAWATCH_CORS_ORIGIN", "*")}})
ALLOWED = {".png", ".jpg", ".jpeg", ".webp"}
CLASS_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

@app.get("/")
def root():
    return jsonify({"service": "RetinaWatch API", "status": "ok", "health": "/api/health"})

@app.get("/api/health")
def health():
    try:
        _model, ckpt = load_model()
        return jsonify({"status": "ok", "model_available": True, "model_version": ckpt.get("architecture", "RetinaWatchCNN-v1")})
    except Exception as exc:
        app.logger.exception("Model not ready")
        return jsonify({"status": "degraded", "model_available": False, "error": str(exc)}), 503

@app.get("/api/model-info")
def model_info():
    metrics_path = ROOT / "reports" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try: metrics = json.loads(metrics_path.read_text())
        except Exception: metrics = {}
    try:
        _model, ckpt = load_model()
        version, best_epoch = ckpt.get("architecture", "RetinaWatchCNN-v1"), ckpt.get("best_epoch")
    except Exception:
        version, best_epoch = None, None
    summary_keys = ["total_images", "train_images", "validation_images", "test_images", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_quadratic_weighted_kappa"]
    return jsonify({
        "project": "RetinaWatch", "model_version": version, "best_epoch": best_epoch,
        "classes": [{"name": n, "grade": i} for i, n in enumerate(CLASS_ORDER)],
        "training_summary": {k: metrics.get(k) for k in summary_keys if k in metrics},
        "limitation": "Academic research prototype only; not clinically validated and not for diagnosis or treatment decisions."
    })

@app.post("/api/predict")
def predict():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose a retinal image to analyse."}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": "Unsupported file type. Upload a PNG, JPG, JPEG or WEBP image."}), 415
    try:
        raw = uploaded.read()
        if not raw:
            return jsonify({"error": "The uploaded file is empty."}), 400
        if len(raw) > 8 * 1024 * 1024:
            return jsonify({"error": "Image is too large. Maximum upload size is 8 MB."}), 413
        image = Image.open(BytesIO(raw))
        image.verify()
        image = Image.open(BytesIO(raw))
        image.load()
        if image.width < 32 or image.height < 32:
            return jsonify({"error": "Image dimensions are too small to analyse."}), 400
        result = predict_with_explanation(image)
        result["filename"] = Path(uploaded.filename).name[:160]
        return jsonify(result)
    except UnidentifiedImageError:
        return jsonify({"error": "This file could not be decoded as an image."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check that the file is a valid retinal image and try again."}), 500

@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"error": "Image is too large. Maximum upload size is 8 MB."}), 413

@app.errorhandler(404)
def not_found(_exc):
    return jsonify({"error": "API route not found."}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
