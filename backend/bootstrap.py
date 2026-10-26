from pathlib import Path
import urllib.request
import sys

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
DATASET = DATA_DIR / "fake_reviews.csv"
DATASET_URL = "https://huggingface.co/datasets/debojit01/fake-review-dataset/resolve/main/fake_reviews.csv?download=true"


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET.exists() and DATASET.stat().st_size > 1_000_000:
        print(f"Dataset already present: {DATASET}")
        return
    print("Downloading the labelled review dataset from Hugging Face...")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "TrustLens-AI/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, DATASET.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if DATASET.stat().st_size < 1_000_000:
        DATASET.unlink(missing_ok=True)
        raise RuntimeError("The dataset download was incomplete.")
    print(f"Dataset downloaded: {DATASET.stat().st_size:,} bytes")


def main():
    ensure_dataset()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core import train_model
    metrics = train_model(DATASET)
    print("Model training completed:")
    print(metrics)


if __name__ == "__main__":
    main()