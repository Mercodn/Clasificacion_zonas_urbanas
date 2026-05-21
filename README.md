# LightScan — Night Light Pollution Classification

A CRISP-ML machine learning project to analyze and classify nighttime light pollution levels in Cundinamarca, Colombia.

## Project Structure

- `app.py` — Flask application with English navigation, CRISP-ML pages, and API endpoints.
- `model.py` — Dataset loading, preprocessing, feature engineering, and classification.
- `scripts/preprocess_balanced.py` — Converts the balanced polygon dataset into point-level data and merges it with VIIRS observations.
- `data.csv` — Original VIIRS dataset.
- `data_prepared.csv` — Prepared dataset used by the model.
- `templates/` — HTML pages for Home, Methodology, Business Understanding, Data Understanding, and Data Engineering.
- `docs/` — English project documentation for the first CRISP-ML phases.
- `Procfile` — Render startup configuration.
- `.gitignore` — Ignored files to keep the repository clean.

## Requirements

Create and activate the Python virtual environment, then install dependencies from `requirements.txt`.

```bash
python -m venv .venv_local
.\.venv_local\Scripts\activate
pip install -r requirements.txt
```

## Run locally

Use the included batch file on Windows:

```bat
cd "C:\Users\Leo\Documents\ML PROYECTO\Clasificacion_zonas_urbanas"
run.bat
```

This will start the app locally on `http://127.0.0.1:5000`.

## Deploy to Render

This repository is ready for Render deployment.

- `Procfile` launches the web process with `gunicorn app:app`.
- `requirements.txt` lists all Python dependencies.
- Render will set the `PORT` environment variable automatically.

If you deploy manually, choose a Python web service and use the default branch.

## Flask pages

- Home
- CRISP-ML Methodology
- Business Understanding
- Data Understanding
- Data Engineering

## Notes

The project is configured for immediate Render deployment and local development using `.venv_local`.
