"""Compact RetinaWatch v3 model inference and Grad-CAM overlay."""
from pathlib import Path
from functools import lru_cache
import base64, io
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).resolve().parent / "models" / "retinawatch_model.pt"
CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
DESCRIPTIONS = {
    "No_DR": "No diabetic-retinopathy pattern detected by this model",
    "Mild": "Mild diabetic-retinopathy pattern",
    "Moderate": "Moderate diabetic-retinopathy pattern",
    "Severe": "Severe diabetic-retinopathy pattern",
    "Proliferate_DR": "Proliferative diabetic-retinopathy pattern",
}
SIZE = 192
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)
torch.set_num_threads(1)

class ConvBNAct(nn.Sequential):
    def __init__(self, ci, co, k=3, s=1, g=1):
        super().__init__(nn.Conv2d(ci, co, k, s, k // 2, groups=g, bias=False),
                         nn.BatchNorm2d(co), nn.Hardswish(inplace=True))

class DS(nn.Module):
    def __init__(self, ci, co, s=1, e=2):
        super().__init__()
        h = max(ci, int(ci * e))
        layers = []
        if h != ci:
            layers.append(ConvBNAct(ci, h, 1))
        layers += [ConvBNAct(h, h, 3, s, h), ConvBNAct(h, co, 1)]
        self.block = nn.Sequential(*layers)
        self.skip = s == 1 and ci == co
    def forward(self, x):
        y = self.block(x)
        return x + y if self.skip else y

class RetinaWatchCNN(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.features = nn.Sequential(ConvBNAct(3, 24, 3, 2), DS(24, 32, 2), DS(32, 48, 2),
                                      DS(48, 64, 2), DS(64, 96, 2), DS(96, 128, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(.30)
        self.classifier = nn.Linear(128, n)
    def forward(self, x):
        return self.classifier(self.dropout(self.pool(self.features(x)).flatten(1)))

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("RetinaWatch model weights are missing")
    ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state, scales = ck["state_dict"], ck.get("quantization")
    if scales:
        state = {k: (v.float() * float(scales[k]) if k in scales else v) for k, v in state.items()}
    model = RetinaWatchCNN(len(CLASSES))
    model.load_state_dict(state)
    model.eval()
    return model, ck

def uri(im):
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")

def quality(im):
    g = np.asarray(im.convert("L").resize((112, 112)), dtype=np.float32)
    lap = -4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:]
    sharp = float(lap.var()) if lap.size else 0.0
    bright, con = float(g.mean()), float(g.std())
    warnings = []
    if min(im.size) < 128: warnings.append("Image resolution is low; fine details may be insufficient.")
    if sharp < 35: warnings.append("Image may be blurry or lack retinal detail.")
    if bright < 12: warnings.append("Image is very dark; useful details may not be visible.")
    elif bright > 245: warnings.append("Image is very bright; exposure may obscure details.")
    if con < 8: warnings.append("Image has very low contrast; the result may be unreliable.")
    return {"warnings": warnings, "blur_score": round(sharp, 1), "mean_brightness": round(bright, 1),
            "contrast_score": round(con, 1), "width": im.width, "height": im.height}

def predict_image(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    display = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(display, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
    model, ck = load_model()
    acts, grads = {}, {}
    def capture(_m, _i, out):
        acts["a"] = out
        out.register_hook(lambda grad: grads.__setitem__("g", grad))
    handle = model.features[5].register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        logits[0, idx].backward()
    handle.remove()
    weights = grads["g"].detach().mean((2, 3), keepdim=True)
    cam = F.relu((weights * acts["a"].detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else np.zeros_like(cam)
    h = np.clip(cam, 0, 1)
    colors = (np.stack([np.clip(1.5-np.abs(4*h-3),0,1), np.clip(1.5-np.abs(4*h-2),0,1),
                        np.clip(1.5-np.abs(4*h-1),0,1)], -1) * 255).astype(np.uint8)
    heat = Image.fromarray(colors).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    overlay = Image.blend(display, heat, .40)
    p = probs.detach().numpy()
    ranked = sorted([{"label": CLASSES[i], "percent": round(float(p[i])*100, 2)} for i in range(5)],
                    key=lambda z: z["percent"], reverse=True)
    confidence = float(p[idx]) * 100
    uncertain = confidence < 55 or ranked[0]["percent"] - ranked[1]["percent"] < 16
    q = quality(im)
    uncertain = uncertain or bool(q["warnings"])
    warnings = list(q["warnings"])
    if uncertain: warnings.insert(0, "The model result is uncertain. Do not rely on this output; seek professional assessment.")
    if CLASSES[idx] in {"Severe", "Proliferate_DR"}:
        warnings.append("Advanced disease classes were weakly detected in internal testing. Seek professional eye-care assessment for health concerns.")
    model.zero_grad(set_to_none=True)
    return {"prediction": CLASSES[idx], "description": DESCRIPTIONS[CLASSES[idx]],
            "confidence": round(confidence, 1), "uncertain": bool(uncertain),
            "uncertainty_message": "The model result is uncertain. Please seek professional assessment." if uncertain else None,
            "probabilities": ranked, "quality": q, "heatmap": uri(overlay), "heatmap_only": uri(heat),
            "warnings": warnings, "model_version": ck.get("architecture", "RetinaWatch-DS-v3"),
            "disclaimer": "Research prototype only. This prediction is not a diagnosis and must not be used to make treatment decisions."}
