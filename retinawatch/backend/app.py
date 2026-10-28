from __future__ import annotations
import json, os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from model import load_model, predict_image, CLASSES

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD = 10 * 1024 * 1024
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/api/health")
def health():
    try:
        load_model()
        return jsonify(status="ok", model_available=True, classes=CLASSES,
                       model_version="RetinaWatch-DS-v3", intended_use="academic prototype only")
    except Exception:
        app.logger.exception("Model failed to load")
        return jsonify(status="degraded", model_available=False), 503

@app.get("/api/methodology")
def methodology():
    try:
        metrics = json.loads((ROOT / "docs" / "metrics.json").read_text())
    except Exception:
        metrics = None
    return jsonify(classes=CLASSES, model_type="Compact depthwise-separable CNN trained from scratch",
                   test_metrics=metrics, clinical_notice="Research prototype only; not for diagnosis or treatment decisions.",
                   explanation="Grad-CAM-style heatmap indicates model sensitivity; it is not validated lesion localization.")

@app.post("/api/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a retinal image to analyse."), 400
    if Path(upload.filename).suffix.lower() not in ALLOWED:
        return jsonify(error="Supported formats: PNG, JPG, JPEG, WEBP or BMP."), 400
    try:
        image = Image.open(upload.stream)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error="The selected file is not a readable image. Choose a valid retinal photograph."), 400
    if image.width * image.height > 20_000_000:
        return jsonify(error="Image dimensions are too large. Resize it and try again."), 400
    if min(image.size) < 64:
        return jsonify(error="Image is too small. Please upload an image at least 64 × 64 pixels."), 400
    try:
        return jsonify(predict_image(image))
    except FileNotFoundError:
        app.logger.exception("Checkpoint missing")
        return jsonify(error="Model weights are not installed."), 503
    except Exception:
        app.logger.exception("Inference failed")
        return jsonify(error="Could not analyse this image. Please try another clear retinal photograph."), 500

@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Image exceeds the 10 MB upload limit."), 413

@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="API endpoint not found."), 404
    return "Page not found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
