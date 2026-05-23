# LightScan — Night Light Pollution Classification

Repository: https://github.com/Mercodn/Clasificacion_zonas_urbanas

A CRISP-ML machine learning project to analyze and classify nighttime light pollution levels in Cundinamarca, Colombia using VIIRS/SUOMI-NPP satellite data.

## Project Overview

This project implements a structured machine learning approach (CRISP-ML) to understand and classify light pollution zones. It combines satellite radiance data, geospatial analysis, and a Gradient Boosting classifier to provide actionable insights for environmental monitoring and urban planning.

## CRISP-ML Phases

### Phase 1: Business & Data Understanding
- **Business Context:** Analyze nighttime light pollution impact on ecosystems, astronomy, and human health
- **Business Objectives:** 
  - Measure and map radiance distribution across Cundinamarca
  - Classify locations into six pollution zones (pristine to metropolitan)
  - Enable scalable prediction API for decision-making
- **Key Questions:** How does pollution vary across urban/rural zones? Which areas have pristine skies? Can we predict zones from radiance alone?
- **Feasibility:** Data available (8,360+ VIIRS observations), standard ML techniques applicable, achievable timeline
- **Risks:** Radiance-only proxy may miss nuances; threshold-based zones may have boundary ambiguities; limited to Cundinamarca
- **Expected Impact:** Environmental baseline, urban planning insights, astronomer resource

### Phase 2: Data Engineering & Modeling
- **Data Cleaning:** Parsed GeoJSON to lat/lon; validated bounds (0–500 nW/cm²/sr)
- **Null Handling:** No missing values (100% complete dataset)
- **Feature Engineering:**
  - `avg_rad` — Raw radiance (primary signal)
  - `log_rad` — Log-transformed (reduces outlier impact)
  - `dist_bogota` — Distance to Bogotá (geographic context)
  - `rad_sq` — Squared radiance (non-linear effects)
  - `lat, lon` — Geographic coordinates
- **Zoning Strategy:** Six radiance-based classes (Bortle scale adapted for satellites)
  - Zone 0: < 0.20 (Pristine)
  - Zone 1: 0.20–0.40 (Rural Low)
  - Zone 2: 0.40–1.00 (Suburban)
  - Zone 3: 1.00–5.00 (Urban Moderate)
  - Zone 4: 5.00–20.0 (Urban High)
  - Zone 5: ≥ 20.0 (Metropolitan)
- **Models:** Gradient Boosting, Random Forest, Logistic Regression
- **Validation:** 80/20 stratified train/test split

### Phase 3: Evaluation & Deployment
- **Performance Metrics:**
  - Accuracy: 100.0% on test set (1672 test samples)
  - ROC curve analysis for multi-class discrimination
  - Confusion matrix for zone-specific errors
- **API Deployment:** Flask + Gunicorn on Render.com
- **Prediction Endpoint:** POST /predict with lat, lon, avg_rad

## Project Structure

```
Clasificacion_zonas_urbanas/
├── app.py                      # Flask application with CRISP-ML routes
├── model.py                    # ML model, preprocessing, predictions
├── data_prepared.csv           # 8,360 VIIRS observations (clean)
├── data.csv                    # Original dataset
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment config
├── README.md                   # This file
├── run.bat                     # Windows startup script
├── templates/                  # HTML pages
│   ├── navbar.html             # Navigation menu
│   ├── index.html              # Home page
│   ├── methodology.html        # CRISP-ML overview
│   ├── business_understanding.html  # Phase 1
│   ├── data_understanding.html      # Phase 1 (EDA)
│   ├── fase2_modelo.html           # Phase 2 (Engineering)
│   └── fase3_evaluacion.html       # Phase 3 (Evaluation)
├── static/                     # CSS, JS, images
│   ├── css/styles.css
│   └── js/main.js
└── docs/                       # Project documentation
    └── data_preparation.md     # Preprocessing details
```

## Dataset

- **Source:** VIIRS/SUOMI-NPP satellite (Nighttime Lights)
- **Records:** 8,360 observations
- **Variables:** avg_rad (radiance in nW/cm²/sr), .geo (GeoJSON coordinates)
- **Geographic Extent:** Cundinamarca, Colombia (3.6–5.9°N, -75.0–-72.9°W)
- **Data Quality:** 100% complete, valid coordinates, realistic radiance values
- **Distribution:** Right-skewed (more rural than urban zones)

## Requirements

Create and activate the Python virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Dependencies include:
- Flask (3.1.3) — Web framework
- scikit-learn (1.8.0) — ML algorithms
- pandas (3.0.3) — Data manipulation
- matplotlib (3.10.9) — Visualization
- numpy (2.4.4) — Numerical computing
- gunicorn (26.0.0) — Production server

## Run Locally

### Windows
```bash
run.bat
```
Opens at `http://127.0.0.1:5000`

### Linux/Mac
```bash
python app.py
```
Opens at `http://127.0.0.1:5000`

## Deploy to Render

1. Push repository to GitHub
2. Create new Render Web Service
3. Connect GitHub repository
4. Set environment: Python 3.11
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn app:app`
7. Deploy

The application will be available at your Render public URL.

## Flask Pages & Routes

| Page | Route | CRISP Phase | Content |
|------|-------|-------------|---------|
| Home | `/` | Overview | Project snapshot, metrics, zone distribution |
| Methodology | `/methodology` | Foundation | CRISP-ML explanation and project coverage |
| Business Understanding | `/business` | Phase 1 | Problem, objectives, feasibility, risks, impact |
| Data Understanding | `/data-understanding` | Phase 1 | EDA, statistics, data quality, distributions |
| Data Engineering | `/data-engineering` | Phase 2 | Feature engineering, transformations, model config |
| Model Engineering | `/model-engineering` | Phase 2 | Multi-model comparison, training diagnostics, prediction examples |
| Evaluation | `/evaluation` | Phase 3 | ROC curve, performance metrics, heatmap |

## API Endpoints

### POST /predict
Classify a light pollution zone for a single location.

**Request:**
```json
{
  "lat": 4.711,
  "lon": -74.072,
  "avg_rad": 15.5
}
```

**Response:**
```json
{
  "zone_id": 4,
  "zone_name": "Urban High",
  "zone_color": "#FFB300",
  "zone_desc": "Dense urban center. Orange sky. High light pollution.",
  "confidence": 87.3,
  "probabilities": {
    "Pristine Natural": 0.1,
    "Rural Low": 0.5,
    "Suburban": 0.2,
    "Urban Moderate": 0.05,
    "Urban High": 0.15,
    "Metropolitan": 0.0
  }
}
```

### GET /api/stats
Global dataset and model statistics.

### GET /api/data
Sample of 500 points for interactive map.

## Results & Insights

- **Accuracy:** {{ stats.accuracy if stats else 'TBD' }}% on stratified test set
- **Top Feature:** avg_rad explains majority of variance
- **Zone Distribution:** Majority of observations in low–medium radiance zones (rural areas)
- **ROC Analysis:** All six zones show strong discrimination (AUC > 0.80 per class)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness: 100.0% on stratified test set |
| **Precision** | Per-zone: True positives / (True pos + False pos) |
| **Recall** | Per-zone: True positives / (True pos + False neg) |
| **ROC Curve** | False Positive Rate vs True Positive Rate per zone |
| **Confusion Matrix** | Detailed zone-to-zone misclassifications |

## Technology Stack

- **Backend:** Flask, Python 3.11
- **ML Framework:** scikit-learn (Gradient Boosting)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, Leaflet.js
- **Frontend:** Bootstrap 5.3, HTML5, CSS3
- **Deployment:** Render.com, Gunicorn
- **Version Control:** GitHub
- **ML Models:** Gradient Boosting, Random Forest, Logistic Regression

## Documentation

Additional documentation available in:
- `docs/data_preparation.md` — Detailed preprocessing workflow
- Flask pages — Interactive CRISP-ML journey through the project
- Inline code comments — Implementation details
- `Submission_Format.md` — Report template for PDF submission

## Notes

- The project is configured for immediate Render deployment
- All data transformations are reproducible and documented
- The model achieves {{ stats.accuracy if stats else 'high' }} accuracy on the test set
- Geographic validation ensures predictions are within Cundinamarca bounds

