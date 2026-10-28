# RetinaWatch

**Explainable diabetic-retinopathy severity research prototype.** Upload a retinal fundus photograph to receive a five-class model estimate, a Grad-CAM overlay, simple image-quality heuristics, and uncertainty flags.

> **Medical safety:** This is an academic demonstration trained on a small, imbalanced, single-source dataset. It is not a clinically validated device and must not be used to diagnose, rule out disease, choose treatment, or replace an eye-care professional.

## Dataset and cleaning

The uploaded archive contains 3,662 PNG images in five class folders: No_DR, Mild, Moderate, Severe, and Proliferate_DR. The folder structure and 224×224 image dimensions match the publicly listed [Diabetic Retinopathy 224×224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) dataset by Sovit Ranjan Rath, described as CC0/Public Domain and acknowledging APTOS 2019. See also [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection). Confirm applicable terms before redistributing any source images. No retinal images are committed to this repository.

Exact duplicate audit found 30 image-hash groups with conflicting labels; those groups were excluded. One copy was kept from each same-label duplicate group. The final 3,504 unique, non-conflicting images were split by stratified sampling into 2,452 training, 526 validation and 526 held-out test images. This is a single-source internal split, not external validation.

## Model and features

- Five-class CNN trained from scratch, with augmentation and class-weighted loss.
- Upload one PNG, JPG, JPEG or WEBP image (max 8 MB).
- Five class scores, uncertainty flag and simple image-quality checks.
- Grad-CAM heatmap and overlay to inspect which regions influenced the model score.
- Held-out metrics served at `GET /api/model-info`.
- Model weights are per-tensor int8-quantised, gzipped and checksummed; the build script reconstructs the compact checkpoint.

No pretrained weights were available in the training environment, so this version was trained from scratch. The model's score is not a calibrated probability. Grad-CAM is not a validated lesion map.

## Held-out test evaluation

Metrics below were computed from the quantised checkpoint on the same 526-image held-out test split. They are dataset-specific and must not be interpreted as clinical performance.

| Metric | Result |
|---|---:|
| Accuracy | 70.34% |
| Balanced accuracy | 55.59% |
| Macro F1 | 51.03% |
| Quadratic weighted kappa | 0.7081 |

| Grade | Test images | Recall | F1 |
|---|---:|---:|---:|
| No_DR | 270 | 94.44% | 94.44% |
| Mild | 51 | 60.78% | 46.97% |
| Moderate | 138 | 43.48% | 53.33% |
| Severe | 27 | 59.26% | 37.21% |
| Proliferate_DR | 40 | 20.00% | 23.19% |

**Major limitation:** The model missed 32 of 40 Proliferate_DR examples in the held-out set (recall 20%). It must not be used to rule out disease or to make healthcare decisions. The poor minority-grade performance is disclosed deliberately and the UI shows strong warnings, not assurances.

## Repository structure

- `backend/app.py` — Flask REST API
- `backend/model.py` — model inference, Grad-CAM and image-quality heuristics
- `backend/models/weight_parts/` — compressed, checksummed model artifact parts
- `backend/bootstrap.py` — reconstructs the model artifact during build
- `frontend/index.html` — responsive single-file web application
- `train.py` — reproducible training/evaluation pipeline
- `reports/metrics.json` — held-out test metrics

## Local development

Place the five extracted class folders under `data/` (for example, `data/No_DR/*.png` and `data/Mild/*.png`), then run:

```bash
python -m pip install -r requirements-train.txt
python train.py
python backend/bootstrap.py
python backend/app.py
```

For production, the backend requires `backend/requirements.txt` and runs with Gunicorn. The frontend is static and its API base URL is currently set to the Render API URL in the HTML. Configure that value for local use if needed.

## API routes

- `GET /api/health` — service and model health
- `GET /api/model-info` — model card and held-out evaluation metrics
- `POST /api/predict` — multipart image upload; form field name `image`

## Limitations and responsible use

- This is a student academic prototype, not a medical device and not clinically validated.
- Do not use it to diagnose, rule out diabetic retinopathy, choose treatment or delay professional care.
- It may be wrong, with particularly poor detection of the Proliferate_DR grade.
- Softmax scores are not calibrated probabilities.
- Grad-CAM is a model-debugging visualisation, not proof of disease or lesion localisation.
- Image-quality flags are heuristic and can miss unsuitable images.
- Test data is from the same source and preprocessing pipeline; performance on other cameras, clinics and populations is unknown.
