"""Decode the small model artifact from base64 gzip parts during deployment."""
from pathlib import Path
import base64, gzip, hashlib, json, sys

HERE = Path(__file__).resolve().parent
PARTS_DIR = HERE / "models" / "weight_parts"
MODEL_PATH = HERE / "models" / "retinawatch_model.pt"
MANIFEST_PATH = PARTS_DIR / "manifest.json"

def main():
    if MODEL_PATH.exists():
        print(f"Model artifact already exists: {MODEL_PATH.name}")
        return
    parts = sorted(PARTS_DIR.glob("weights-*.b64"))
    if not parts:
        print("Model weights are not bundled. Expected models/weight_parts/weights-*.b64", file=sys.stderr)
        sys.exit(2)
    encoded = "".join(p.read_text().strip() for p in parts)
    compressed = base64.b64decode(encoded, validate=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        expected = manifest.get("gzip_sha256")
        actual = hashlib.sha256(compressed).hexdigest()
        if expected and actual != expected:
            raise ValueError("Model artifact checksum mismatch; refusing to unpack weights.")
    raw = gzip.decompress(compressed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(raw)
    print(f"Decoded model weights: {MODEL_PATH.name} ({len(raw):,} bytes); source parts: {len(parts)}")

if __name__ == "__main__":
    main()
