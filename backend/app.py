from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import os
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import analyse_reviews, load_dataset, get_model

app = Flask(__name__)
allowed_origins = os.environ.get("TRUSTLENS_CORS_ORIGIN", "*")
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED = {".csv", ".xlsx", ".xls"}


@app.get("/")
def root():
    return jsonify({"service": "TrustLens AI API", "status": "ok", "health": "/api/health"})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_available": bool(get_model())})


@app.get("/api/dataset-summary")
def dataset_summary():
    try:
        df = load_dataset()
        return jsonify({
            "rows": int(len(df)),
            "columns": list(df.columns),
            "labels": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
            "categories": int(df["category"].nunique()) if "category" in df else 0,
            "ratings": {str(k): int(v) for k, v in df["rating"].value_counts(dropna=True).sort_index().items()} if "rating" in df else {}
        })
    except Exception as exc:
        app.logger.exception("Could not summarize dataset")
        return jsonify({"error": str(exc)}), 503


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    reviews = data.get("reviews", [])
    if isinstance(reviews, str):
        reviews = [{"text": line} for line in reviews.splitlines() if line.strip()]
    if not isinstance(reviews, list):
        return jsonify({"error": "reviews must be a list or newline-separated string"}), 400
    if len(reviews) > 5000:
        return jsonify({"error": "Please submit no more than 5,000 reviews at a time."}), 400
    normalized = []
    for item in reviews:
        if isinstance(item, str):
            normalized.append({"text": item})
        elif isinstance(item, dict):
            normalized.append({
                "text": item.get("text", item.get("text_", "")),
                "rating": item.get("rating"),
                "product": item.get("product", item.get("category", ""))
            })
    result = analyse_reviews(normalized)
    if result["total"] == 0:
        return jsonify({"error": "Add at least one non-empty review."}), 400
    return jsonify(result)


@app.post("/api/analyze-file")
def analyze_file():
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "Choose a CSV or Excel file."}), 400
    ext = Path(uploaded.filename or "").suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": "Supported formats are CSV, XLSX and XLS."}), 400
    try:
        frame = pd.read_csv(uploaded) if ext == ".csv" else pd.read_excel(uploaded)
        columns = {str(c).lower().strip(): c for c in frame.columns}
        text_column = next((columns[c] for c in ["text_", "text", "review", "review_text", "content", "body"] if c in columns), None)
        if text_column is None:
            return jsonify({"error": "No review text column found. Use text_, text, review, review_text, content, or body."}), 400
        rating_column = next((columns[c] for c in ["rating", "stars", "score"] if c in columns), None)
        product_column = next((columns[c] for c in ["product", "product_name", "category"] if c in columns), None)
        normalized = []
        for _, row in frame.head(5000).iterrows():
            normalized.append({
                "text": row[text_column],
                "rating": row[rating_column] if rating_column else None,
                "product": row[product_column] if product_column else ""
            })
        result = analyse_reviews(normalized)
        result["source_filename"] = uploaded.filename
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Could not process uploaded file")
        return jsonify({"error": f"Unable to read this file: {exc}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)