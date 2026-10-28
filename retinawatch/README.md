# RetinaWatch

RetinaWatch is a student research prototype for five-class diabetic-retinopathy grading from retinal fundus images. It returns a model-estimated class, class scores, a Grad-CAM-style overlay, basic image-quality warnings, and an uncertainty flag.

## Evaluation
The supplied archive contained 3,662 labeled images. After duplicate review, 3,504 unique non-conflicting images were retained; 30 conflicting-label duplicate groups were excluded and 96 same-label copies collapsed. On a held-out split of 526 images, the compact CNN reached 70.15% accuracy, 56.40% balanced accuracy, 0.516 macro F1 and 0.717 quadratic weighted kappa. Per-class performance varies substantially; advanced classes were weakly detected. See docs/metrics.json.

## Safety
This is a research prototype, not a medical device. Do not use it for diagnosis, screening decisions, or treatment decisions. The heatmap is not validated lesion localization, and model scores are not calibrated clinical probabilities. Use a qualified eye-care professional for health concerns.

## Render deployment
Build command: `pip install -r retinawatch/backend/requirements.txt && python retinawatch/bootstrap.py`
Start command: `gunicorn --chdir retinawatch/backend app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`

The original dataset is not included in the repository. Confirm dataset provenance and license before redistribution.