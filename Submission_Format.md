# Submission Format

This document is a ready-to-use report template for the project submission PDF.

## 1. Model Descriptions

Describe each implemented model and its purpose.

- **Gradient Boosting**
  - Description: Tree boosting algorithm that captures non-linear relationships between radiance, geography, and zone labels.
  - Hyperparameters: `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`.
  - Role: Best-performing model selected for production.

- **Random Forest**
  - Description: Bagged ensemble of decision trees for robustness and reduced variance.
  - Hyperparameters: `n_estimators=150`, `max_depth=10`, `random_state=42`, `n_jobs=-1`.
  - Role: Robust baseline that handles spatial and radiance features.

- **Logistic Regression**
  - Description: Linear classification baseline for multi-class prediction.
  - Hyperparameters: `solver='lbfgs'`, `max_iter=1200`, `random_state=42`.
  - Role: Provides an interpretable baseline and validates the need for non-linear models.

## 2. Training Process

Explain the dataset split, preprocessing, feature engineering, and training workflow.

- Data loaded from `data_prepared.csv`.
- Target classes generated from `avg_rad` using defined zone thresholds.
- Engineering of features:
  - `avg_rad` — raw radiance
  - `log_rad` — log-transformed radiance
  - `dist_bogota` — Euclidean distance to Bogotá
  - `rad_sq` — squared radiance
  - `lat`, `lon` — geographic coordinates
- Dataset split: 80% train, 20% test with stratification.
- Scaling: `StandardScaler` fit on training data and applied to test data.
- Each model trained on the same training split, then evaluated on the same held-out test split.
- Model selection based on highest test accuracy.

### Training Code Example

```python
from model import LightPollutionClassifier

classifier = LightPollutionClassifier()
accuracy = classifier.train()
print(f"Best model: {classifier.best_model_name}")
print(f"Test accuracy: {accuracy:.1f}%")
```

## 3. Validation Process

Document how model performance was validated.

- Metrics computed on the held-out test set:
  - Accuracy
  - Precision (macro average)
  - Recall (macro average)
  - F1 score (macro average)
- Visualizations included in the app:
  - ROC curve for the selected model
  - Confusion matrix for test predictions
  - Spatial heatmap of predicted zones
- Comparison table in the Model Engineering page shows all three models side by side.

## 4. Console Predictions

Show command-line and Python examples for requesting predictions.

### cURL example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": 4.711, "lon": -74.072, "avg_rad": 15.5}'
```

### Python example

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"lat": 4.711, "lon": -74.072, "avg_rad": 15.5}
)
print(response.json())
```

## 5. Flask UI Screenshots

Include screenshots of:

- Home page
- CRISP-ML Methodology page
- Data Engineering / Model Development page
- Model Engineering comparison page
- Model Evaluation page

Tip: Use the browser screenshot tool or OS screenshot utility and paste the images into the PDF report.

## 6. Repository and Deployment

Include the repository and deployment URLs.

- GitHub repository: `https://github.com/Mercodn/Clasificacion_zonas_urbanas`
- Render URL: `https://<your-service>.onrender.com`

## 7. Submission Checklist

- [ ] Model descriptions included
- [ ] Training process explained
- [ ] Validation process documented
- [ ] Console prediction examples present
- [ ] Screenshots included
- [ ] GitHub repository updated
- [ ] Render URL included

---

This file can be exported to PDF using any markdown-to-PDF converter or pasted into a document editor.
