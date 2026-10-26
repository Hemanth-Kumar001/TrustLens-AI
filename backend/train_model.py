from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import train_model

if __name__ == "__main__":
    metrics = train_model()
    print("Training complete.")
    print(metrics)