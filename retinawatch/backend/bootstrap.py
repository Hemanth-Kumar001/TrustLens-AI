"""Decode and verify bundled quantized RetinaWatch weights during deployment."""
from pathlib import Path
import base64, gzip, hashlib, json, sys

HERE = Path(__file__).resolve().parent
PARTS_DIR = HERE / "models" / "weight_parts"
MODEL_PATH = HERE / "models" / "retinawatch_model.pt"
MANIFEST_PATH = PARTS_DIR / "manifest.json"

def git_blob_sha1(text: str) -> str:
    data = text.encode("ascii")
    header = f"blob {len(data)}\\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()

def main():
    if MODEL_PATH.exists():
        print(f"Model artifact already exists: {MODEL_PATH.name}")
        return
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing weight manifest; refusing to load unchecked model parts.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected_parts = manifest.get("part_lengths", {})
    expected_hashes = manifest.get("part_sha1", {})
    parts = sorted(PARTS_DIR.glob("weights-*.b64"))
    if not parts or [p.name for p in parts] != sorted(expected_parts):
        raise ValueError("Model parts are missing or do not match the manifest.")
    contents = [p.read_text().strip() for p in parts]
    for index, (part, value) in enumerate(zip(parts, contents)):
        target_len = int(expected_parts[part.name])
        target_sha = expected_hashes.get(part.name)
        if len(value) == target_len and git_blob_sha1(value) == target_sha:
            continue
        # A single text-transfer may accidentally insert one base64 character.
        # Only attempt recovery when a known length + SHA-1 manifest proves the result.
        if len(value) != target_len + 1 or not target_sha:
            raise ValueError(f"Model part integrity check failed: {part.name}")
        repaired = None
        for position in range(len(value)):
            candidate = value[:position] + value[position + 1:]
            if git_blob_sha1(candidate) == target_sha:
                repaired = candidate
                break
        if repaired is None:
            raise ValueError(f"Could not safely repair model part: {part.name}")
        contents[index] = repaired
        print(f"Verified one-character transfer repair in {part.name}")
    encoded = "".join(contents)
    if len(encoded) != int(manifest["base64_characters"]) or len(encoded) % 4:
        raise ValueError("Base64 model artifact length does not match the manifest.")
    compressed = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(compressed).hexdigest()
    if actual != manifest["gzip_sha256"]:
        raise ValueError("Compressed model checksum mismatch; refusing to load weights.")
    raw = gzip.decompress(compressed)
    if len(raw) != int(manifest["raw_bytes"]):
        raise ValueError(f"Unexpected decoded model size: {len(raw)} bytes.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(raw)
    print(f"Verified and decoded model weights: {MODEL_PATH.name} ({len(raw):,} bytes); source parts: {len(parts)}")

if __name__ == "__main__":
    main()
