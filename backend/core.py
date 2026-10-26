from pathlib import Path
import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("TRUSTLENS_DATASET", BASE / "data" / "fake_reviews.csv"))
MODEL_PATH = BASE / "models" / "review_classifier.joblib"


def clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def load_dataset(path=DATASET):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run backend/bootstrap.py to download it first."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"text_", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain text_ and label columns.")
    df = df.copy()
    df["text_"] = df["text_"].map(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[(df["text_"].str.len() > 0) & df["label"].isin(["CG", "OR"])]
    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both CG and OR labels.")
    return df


def train_model(path=DATASET):
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50000,
            sublinear_tf=True, strip_accents="unicode", dtype="float32"
        )),
        ("classifier", LogisticRegression(
            max_iter=600, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    metrics = {
        "test_rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": sorted(df["label"].unique().tolist()),
        "trained_rows": int(len(df)),
        "note": "Held-out evaluation on this dataset only; not a real-world guarantee."
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics, "trained_rows": int(len(df))}, MODEL_PATH, compress=3)
    return metrics


def get_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


POSITIVE_WORDS = set("""
excellent amazing love great good perfect comfortable sturdy beautiful easy recommend best
smooth fast reliable useful worth fantastic wonderful impressed satisfied happy excellent quality
""".split())
NEGATIVE_WORDS = set("""
bad poor terrible awful hate broken disappointed uncomfortable worst slow weak difficult issue
problem faulty waste annoying horrible overheat overheating drains drained freezes frozen damaged
disappointed dead pixels failed failure stopped noisy loud terrible unreliable not-working
""".split())

COMPLAINTS = {
    "Battery life": [
        "battery drains", "drains quickly", "battery is terrible", "barely lasts", "short battery",
        "poor battery", "does not last", "doesn't last", "battery life is poor", "battery life is terrible",
        "charge runs out", "battery died"
    ],
    "Heating & fan noise": [
        "overheat", "overheating", "too hot", "gets very hot", "fan noise", "loud fan",
        "fan is very loud", "noisy fan", "gets warm"
    ],
    "Build quality": [
        "cheap plastic", "flimsy", "broke", "broken", "poor build", "build quality is poor",
        "cracked", "dead pixels", "stopped working", "fell apart"
    ],
    "Performance": [
        "slow", "lag", "lagging", "freezes", "freezing", "crash", "crashes", "hangs",
        "freezes randomly", "runs slowly", "keeps restarting"
    ],
    "Display & sound": [
        "dead pixels", "screen flicker", "flickering screen", "dim display", "poor display",
        "crackling sound", "speaker stopped", "screen is blurry", "low brightness"
    ],
    "Delivery & packaging": [
        "delivery was late", "late delivery", "shipping was late", "arrived damaged", "damaged packaging",
        "package was damaged", "packaging was damaged", "delivery arrived"
    ],
    "Customer service": [
        "customer service", "customer support", "support refused", "refund", "return",
        "warranty", "has not helped", "not helped", "ignored my complaint"
    ],
    "Price & value": [
        "too expensive", "overpriced", "not worth the price", "poor value", "waste of money",
        "expected much better quality at this price", "not worth the money"
    ]
}
PRAISE = {
    "Performance": ["fast", "great performance", "smooth", "powerful", "responsive", "runs smoothly", "handles programming"],
    "Battery life": ["lasts long", "long battery", "all day", "battery lasts", "battery life is excellent"],
    "Build & design": ["beautiful design", "beautiful", "sturdy", "premium", "lightweight", "good build quality", "comfortable"],
    "Display & sound": ["excellent display", "bright display", "clear sound", "great speakers", "picture quality"],
    "Value for money": ["value for money", "worth the money", "affordable", "good value", "great value"],
    "Ease of use": ["easy to use", "simple to use", "setup was easy", "convenient", "easy setup"]
}

SEVERE_NEGATIVE = (
    "stopped working", "dead pixels", "freezes randomly", "barely lasts",
    "battery is terrible", "very disappointed", "support has not helped",
    "started overheating", "fan is very loud", "terrible", "awful", "broken",
    "faulty", "failed after", "does not work", "doesn't work", "keeps crashing"
)


def sentiment(text, rating=None):
    low = clean_text(text).lower()
    if any(phrase in low for phrase in SEVERE_NEGATIVE):
        return "Negative"
    try:
        if rating is not None and str(rating).strip() and not pd.isna(float(rating)):
            n = float(rating)
            if n >= 4:
                return "Positive"
            if n <= 2:
                return "Negative"
            return "Neutral"
    except (ValueError, TypeError):
        pass
    words = re.findall(r"[a-z']+", low)
    score = sum(w in POSITIVE_WORDS for w in words) - sum(w in NEGATIVE_WORDS for w in words)
    # Negation should reduce the positive signal in short phrases such as "not good".
    if re.search(r"\b(not|never|no longer)\s+(good|great|excellent|worth|recommend|fast)\b", low):
        score -= 2
    if score >= 1:
        return "Positive"
    if score <= -1:
        return "Negative"
    return "Neutral"


def matched_themes(text, mapping):
    low = clean_text(text).lower()
    return [theme for theme, terms in mapping.items() if any(term in low for term in terms)]


def analyse_reviews(reviews):
    bundle = get_model()
    texts = [clean_text(x.get("text", "")) for x in reviews]
    valid = [i for i, value in enumerate(texts) if value]
    predictions = {}
    if bundle and valid:
        model = bundle["model"]
        predicted = model.predict([texts[i] for i in valid])
        try:
            probabilities = model.predict_proba([texts[i] for i in valid])
        except Exception:
            probabilities = None
        for j, i in enumerate(valid):
            predictions[i] = {
                "label": str(predicted[j]),
                "confidence": round(float(max(probabilities[j])), 3) if probabilities is not None else None
            }
    output = []
    for index, item in enumerate(reviews):
        text = texts[index]
        if not text:
            continue
        pred = predictions.get(index)
        label = ("Computer-generated-like" if pred["label"] == "CG" else "Original-like") if pred else "Model not trained"
        complaints = matched_themes(text, COMPLAINTS)
        praise = matched_themes(text, PRAISE)
        # A topic is never shown in both lists for the same review; this avoids contradictory duplicated tags.
        complaint_topics = {x.lower() for x in complaints}
        praise = [x for x in praise if x.lower() not in complaint_topics]
        output.append({
            "id": len(output) + 1,
            "text": text,
            "rating": item.get("rating"),
            "product": clean_text(item.get("product", "")),
            "sentiment": sentiment(text, item.get("rating")),
            "review_class": label,
            "class_code": pred["label"] if pred else None,
            "confidence": pred["confidence"] if pred else None,
            "complaints": complaints,
            "praise": praise
        })
    total = len(output)
    sentiments = {s: sum(r["sentiment"] == s for r in output) for s in ["Positive", "Neutral", "Negative"]}
    classes = {s: sum(r["class_code"] == s for r in output) for s in ["CG", "OR"]}
    complaint_counts = {k: sum(k in r["complaints"] for r in output) for k in COMPLAINTS}
    praise_counts = {k: sum(k in r["praise"] for r in output) for k in PRAISE}
    complaint_counts = {k: v for k, v in sorted(complaint_counts.items(), key=lambda kv: kv[1], reverse=True) if v}
    praise_counts = {k: v for k, v in sorted(praise_counts.items(), key=lambda kv: kv[1], reverse=True) if v}
    return {
        "total": total,
        "sentiments": sentiments,
        "classes": classes,
        "complaints": [{"theme": k, "count": v, "share": round(v / max(total, 1) * 100, 1)} for k, v in list(complaint_counts.items())[:8]],
        "strengths": [{"theme": k, "count": v, "share": round(v / max(total, 1) * 100, 1)} for k, v in list(praise_counts.items())[:8]],
        "reviews": output,
        "model_metrics": bundle.get("metrics") if bundle else None,
        "model_available": bool(bundle)
    }