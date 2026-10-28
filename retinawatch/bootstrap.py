"""Reconstruct the compressed checkpoint from text-safe GitHub assets."""
from pathlib import Path
import base64, gzip
ROOT = Path(__file__).resolve().parent
parts = sorted((ROOT / "model").glob("weights.part*.b64"), key=lambda p: int(p.stem.replace("weights.part", "")))
out = ROOT / "backend" / "models" / "retinawatch_model.pt"
out.parent.mkdir(parents=True, exist_ok=True)
if len(parts) < 2:
    raise FileNotFoundError("Model checkpoint parts are missing.")
payload = "".join(p.read_text().strip() for p in parts)
out.write_bytes(gzip.decompress(base64.b64decode(payload, validate=True)))
print(f"RetinaWatch model checkpoint ready ({out.stat().st_size:,} bytes)")
