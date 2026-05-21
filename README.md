# LightScan — Night Light Pollution Classification

A CRISP-ML machine learning project to analyze and classify nighttime light pollution levels in Cundinamarca, Colombia.

## Project Structure

- `app.py` — Flask application with English navigation and CRISP-ML pages.
- `model.py` — Dataset loading, preprocessing, feature engineering, and classification.
- `scripts/preprocess_balanced.py` — Converts the balanced polygon dataset into point-level data and merges it with VIIRS observations.
- `data.csv` — Original VIIRS dataset.
- `data_prepared.csv` — Prepared dataset used by the model.
- `templates/` — HTML pages for Home, Methodology, Business Understanding, Data Understanding, and Data Engineering.
- `docs/` — English project documentation for the first CRISP-ML phases.

## Requirements

Create and activate a Python virtual environment, then install dependencies from `requirements.txt`.

```bash
python -m venv .venv_local
.\.venv_local\Scripts\activate
pip install -r requirements.txt
```

## Run locally

If you are on Windows, use the included batch file:

```bat
cd "C:\Users\Leo\Documents\ML PROYECTO\Clasificacion_zonas_urbanas"
run.bat
```

The script will create `.venv_local` and install dependencies if needed, then start the app.

Then visit `http://127.0.0.1:5000`.

## Flask pages

- Home
- CRISP-ML Methodology
- Business Understanding
- Data Understanding
- Data Engineering

## Notes

This project is designed to support the first stages of a machine learning project: business understanding, data understanding, and data engineering.
