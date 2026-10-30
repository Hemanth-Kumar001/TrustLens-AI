# TrustLens AI

🌐 **Live Website:** [Open TrustLens AI](https://trustlens-ai-web.onrender.com) · 💻 **Source Code:** [GitHub Repository](https://github.com/Hemanth-Kumar001/TrustLens-AI)

**Review intelligence for better-informed product decisions.**

TrustLens AI analyses pasted or uploaded product reviews, summarizes sentiment, surfaces recurring complaints and praised themes, and flags text that resembles the classes in a labelled training dataset.

## Features
- React + Vite frontend with a cream editorial landing page and dark analysis dashboard
- Flask API for pasted reviews and CSV/Excel uploads
- TF-IDF text classification with six compared classical ML algorithms (Logistic Regression, Multinomial Naive Bayes, Complement Naive Bayes, SGD Classifier, Decision Tree, and Random Forest)
- Explainable keyword-based complaint and praise themes
- Review evidence, CSV export, and browser-local history
- Render deployment configuration

## Dataset and model notes
The development dataset contains the columns category, rating, label, and text_, with CG (computer-generated) and OR (original) labels. After text cleaning and filtering to these two dataset labels, the reference CSV contains 40,432 usable records. The project uses a stratified 70/15/15 train/validation/test split. Six TF-IDF classifiers are compared on the validation split using accuracy, macro precision, macro recall, and macro F1. The highest validation macro F1 selects the algorithm; it is refitted on train plus validation data and evaluated once on the separate held-out test split. These dataset labels are not proof that an individual real-world review is fraudulent or genuine. Predictions should be treated as indicators only.

For cloud builds, the deployment bootstrap downloads a compatible version of the dataset from the Hugging Face dataset repository debojit01/fake-review-dataset (MIT-labelled dataset card) and trains the model during the build. The model is not committed as a binary artifact. The dataset source and its licence notice are documented in the application.

## Local development
Requires Python 3.12 and Node.js 20.19+.

Backend:
```powershell
py -3.12 -m pip install -r backend/requirements.txt
py -3.12 backend/bootstrap.py
py -3.12 backend/app.py
```

Frontend, in a second terminal:
```powershell
cd frontend
npm install
npm run dev
```

Frontend defaults to http://127.0.0.1:5000/api locally. Set VITE_API_URL at build time to the deployed API URL plus /api.

## Render
This repository includes render.yaml to define a Python API and a static frontend. Connect the repository to Render as a Blueprint and deploy the two services. After Render assigns service URLs, ensure VITE_API_URL points to the API URL ending in /api and TRUSTLENS_CORS_ORIGIN matches the frontend origin exactly.

## Limitations
- Sentiment and complaint/praise themes use transparent heuristics; they are not a fine-tuned sentiment model.
- Arbitrary product URLs, screenshot OCR, hosted accounts/cloud-sync, and PDF export are not included.
- Model metrics and the comparison table reflect this dataset and the stated random split. Results may not generalize to current reviews, other products, or other platforms.
- The six-model comparison is part of the classifier training workflow. It does not turn dataset labels into verified evidence of fraud or authenticity.

## Attribution
Dataset source: https://huggingface.co/datasets/debojit01/fake-review-dataset. Review the source card before redistributing data separately. The dataset is fetched for build-time training rather than stored in this GitHub repository.
## Reference model comparison
A reference run on the 40,432-row cleaned CSV (random state 42) produced the following validation results. Values are percentages; the model is selected by validation macro F1.

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| SGD Classifier | 94.30% | 94.29% |
| Logistic Regression | 93.21% | 93.21% |
| Multinomial Naive Bayes | 91.06% | 91.06% |
| Complement Naive Bayes | 91.06% | 91.06% |
| Random Forest | 87.65% | 87.63% |
| Decision Tree | 70.75% | 70.31% |

**Selected model:** SGD Classifier. On the separate held-out test split, the reference run returned accuracy 94.61%, macro precision 94.63%, macro recall 94.61%, and macro F1 94.61%.

These are results from one reproducible split of this dataset, not guarantees of real-world performance. Re-run the workflow to regenerate the comparison if dependencies or source data change.
