"""RetinaWatch inference and Grad-CAM utilities."""
from pathlib import Path
from functools import lru_cache
import io
import base64
import threading
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).resolve().parent / "models" / "retinawatch_model.pt"
IMAGE_SIZE = 224
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
CLASS_INFO = {
    "No_DR": {"grade": 0, "display": "No diabetic retinopathy", "short": "No DR", "description": "The model's closest class is the no-DR category. This is not a medical clearance."},
    "Mild": {"grade": 1, "display": "Mild non-proliferative DR", "short": "Mild", "description": "The model's closest class is mild non-proliferative diabetic retinopathy."},
    "Moderate": {"grade": 2, "display": "Moderate non-proliferative DR", "short": "Moderate", "description": "The model's closest class is moderate non-proliferative diabetic retinopathy."},
    "Severe": {"grade": 3, "display": "Severe non-proliferative DR", "short": "Severe", "description": "The model's closest class is severe non-proliferative diabetic retinopathy."},
    "Proliferate_DR": {"grade": 4, "display": "Proliferative DR", "short": "Proliferative", "description": "The model's closest class is proliferative diabetic retinopathy."},
}
DEVICE = torch.device("cpu")
INFER_LOCK = threading.Lock()

class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size=kernel, stride=stride, padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout), nn.Hardswish(inplace=True)
        )

class DepthwiseSeparable(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=1):
        super().__init__()
        hidden = max(cin, int(cin * expansion))
        layers = []
        if hidden != cin:
            layers.append(ConvBNAct(cin, hidden, kernel=1))
        layers += [ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden), ConvBNAct(hidden, cout, kernel=1)]
        self.block = nn.Sequential(*layers)
        self.skip = stride == 1 and cin == cout
    def forward(self, x):
        y = self.block(x)
        return y + x if self.skip else y

class RetinaWatchCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, kernel=3, stride=2),
            DepthwiseSeparable(24, 32, stride=2, expansion=2),
            DepthwiseSeparable(32, 48, stride=2, expansion=2),
            DepthwiseSeparable(48, 64, stride=2, expansion=2),
            DepthwiseSeparable(64, 96, stride=2, expansion=2),
            DepthwiseSeparable(96, 128, stride=1, expansion=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = RetinaWatchCNN(num_classes=len(CLASS_NAMES))
    state = checkpoint["state_dict"]
    # Model artifact is stored with symmetric per-tensor int8 quantisation to reduce deploy size.
    scales = checkpoint.get("quantization")
    if scales:
        restored = {}
        for key, value in state.items():
            if key in scales:
                restored[key] = value.to(torch.float32) * float(scales[key])
            else:
                restored[key] = value
        state = restored
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model, checkpoint

def image_to_tensor(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    return TRANSFORM(image).unsqueeze(0)

def _png_data_uri(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def quality_check(image: Image.Image):
    flags = []
    width, height = image.size
    if min(width, height) < 128:
        flags.append("The image resolution is low; detail may be insufficient.")
    gray = np.asarray(image.convert("RGB").resize((112, 112)), dtype=np.float32).mean(axis=2)
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    sharpness = float(lap.var()) if lap.size else 0.0
    brightness = float(gray.mean())
    contrast = float(gray.std())
    if sharpness < 60:
        flags.append("The image may be blurred or low-detail. Consider a sharper fundus photograph.")
    if brightness < 12:
        flags.append("The image is extremely dark; useful retinal details may not be visible.")
    elif brightness > 205:
        flags.append("The image is extremely bright; exposure may obscure retinal details.")
    if contrast < 8:
        flags.append("The image has very low contrast; the model output may be unreliable.")
    return {"flags": flags, "sharpness_score": round(sharpness, 1), "mean_brightness": round(brightness, 1), "contrast_score": round(contrast, 1), "minimum_dimension": min(width, height)}

def predict_with_explanation(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    display_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    tensor = image_to_tensor(display_image).to(DEVICE)
    model, checkpoint = load_model()
    activations, gradients = {}, {}
    def capture(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))
    with INFER_LOCK:
        handle = model.features.register_forward_hook(capture)
        model.zero_grad(set_to_none=True)
        tensor.requires_grad_(True)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        logits[0, predicted_index].backward()
        handle.remove()
        activation, gradient = activations["value"].detach(), gradients["value"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > cam.min() else np.zeros_like(cam)
        probs = probabilities.detach().cpu().numpy()
        model.zero_grad(set_to_none=True)
    class_name = CLASS_NAMES[predicted_index]
    ranked = sorted([{"class_name": name, "label": CLASS_INFO[name]["display"], "grade": CLASS_INFO[name]["grade"], "score": round(float(probs[i]) * 100, 2)} for i, name in enumerate(CLASS_NAMES)], key=lambda item: item["score"], reverse=True)
    margin = float(ranked[0]["score"] - ranked[1]["score"])
    uncertain = float(ranked[0]["score"]) < 50.0 or margin < 12.0
    notes = []
    if uncertain:
        notes.append("The model is uncertain: the top score is low or close to the next class. Do not rely on this prediction.")
    if float(ranked[0]["score"]) > 90:
        notes.append("A high softmax score is not a calibrated probability and does not establish diagnostic certainty.")
    quality = quality_check(image)
    if quality["flags"]:
        notes.append("Image-quality warnings are heuristic and may not identify every unsuitable image.")
    heat = np.clip(cam, 0, 1)
    color = (np.stack([np.clip(1.5 * heat, 0, 1), np.clip(1.6 - np.abs(heat - 0.65) * 3.0, 0, 1), np.clip(1.0 - 1.8 * heat, 0, 1)], axis=-1) * 255).astype(np.uint8)
    heat_image = Image.fromarray(color, mode="RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    overlay = Image.blend(display_image, heat_image, alpha=0.40)
    return {
        "predicted_class": class_name, "predicted_label": CLASS_INFO[class_name]["display"],
        "grade": CLASS_INFO[class_name]["grade"], "description": CLASS_INFO[class_name]["description"],
        "model_score": round(float(probs[predicted_index]) * 100, 2), "second_best_margin": round(margin, 2),
        "uncertain": uncertain, "class_probabilities": ranked, "quality": quality, "warnings": notes,
        "heatmap_data_uri": _png_data_uri(heat_image), "overlay_data_uri": _png_data_uri(overlay),
        "input_image_size": {"width": int(original_size[0]), "height": int(original_size[1])},
        "model_version": checkpoint.get("architecture", "RetinaWatchCNN-v1"),
        "explanation_note": "Grad-CAM highlights image regions that influenced this model score. It is not a validated lesion segmentation or clinical explanation.",
    }
