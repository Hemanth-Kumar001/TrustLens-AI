# RetinaWatch

**Explainable diabetic-retinopathy severity research prototype.** Upload a retinal fundus photograph to receive a five-class model estimate, a Grad-CAM overlay, simple image-quality heuristics, and uncertainty flags.

> **Medical safety:** This is an academic demonstration trained on a small, imbalanced, single-source dataset. It is not a clinically validated device and must not be used to diagnose, rule out disease, choose treatment, or replace an eye-care professional.

## Dataset

The supplied archive contains 3,662 PNG images in five class folders: No_DR, Mild, Moderate, Severe, and Proliferate_DR. The folder structure and 224x224 image dimensions match the publicly listed Diabetic Retinopathy 224x224 (2019 Data) dataset by Sovit Ranjan Rath, which states CC0/Public Domain and acknowledges APTOS 2019 Blindness Detection. See https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data and https://www.kaggle.com/competitions/aptos2019-blindness-detection. Verify applicable dataset terms before redistributing source images. No source images are included here.

## Structure

- backend/app.py — Flask REST API
- backend/model.py — CNN, inference, Grad-CAM and image checks
- backend/models/weight_parts/ — compressed, checksummed model artifact parts
- backend/bootstrap.py — reconstructs the model artifact at build time
- frontend/index.html — responsive single-file web application
- train.py — reproducible training and evaluation pipeline
- reports/metrics.json — held-out test metrics

The original dataset is deliberately excluded from Git. The saved model is included as a compressed int8 artifact and decoded during deployment, so the site does not need to download or retrain on the entire dataset.

## Local development

Place the five extracted class folders directly under data/ (for example data/No_DR/*.png and data/Mild/*.png), then run:

    python -m pip install -r backend/requirements.txt scikit-learn
    python train.py
    python backend/bootstrap.py
    python backend/app.py

Run the static website by opening frontend/index.html or serving that directory. Change API_BASE near the top of its script to your local API URL when running locally.

## API routes

- GET /api/health — service and model health
- GET /api/model-info — model card and training metrics
- POST /api/predict — multipart image upload with field name image

## Evaluation

The held-out test split contains 733 images and is separate from the training and validation split. Because the classes are imbalanced, review accuracy together with balanced accuracy, macro-F1, per-class results and the confusion matrix. The proliferative class is a particularly weak point in this model.

The class score is the highest softmax output and is not a calibrated probability. Grad-CAM highlights regions that influenced the selected model score; it is not validated lesion segmentation. Blur, brightness and contrast messages are basic heuristics, not clinical quality-control tests. Internal test results do not demonstrate performance on new hospitals, cameras, populations or real-world screening.
