# ══════════════════════════════════════════════════════════════
#  app.py  ·  Light Pollution Classifier
#  Flask Application - CRISP-ML(Q) Project
#  Cundinamarca VIIRS / SUOMI-NPP Analysis
# ══════════════════════════════════════════════════════════════

import io
import base64
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from flask import Flask, render_template, request, jsonify

# ── Import ML module ──────────────────────────────────
from model import (
    LightPollutionClassifier,
    ZONE_NAMES,
    ZONE_COLORS,
    GEO_BOUNDS,
    DATA_PATH,
)

# ─────────────────────────────────────────
# INITIALIZE FLASK
# ─────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ─────────────────────────────────────────
# TRAIN MODEL AT STARTUP
# ─────────────────────────────────────────
classifier = LightPollutionClassifier()
classifier.train(DATA_PATH)


# ─────────────────────────────────────────
# VISUALIZATION · HEATMAP
# ─────────────────────────────────────────

_VIIRS_COLORS = [
    "#000000", "#000033", "#000080", "#0040FF",
    "#00CFFF", "#00FF80", "#FFFF00", "#FFB300", "#FF4400", "#FF0000"
]
_VIIRS_CMAP = LinearSegmentedColormap.from_list("viirs", _VIIRS_COLORS, N=512)


def _render_heatmap(df, highlight_point: dict = None) -> str:
    """Generate VIIRS heatmap as base64-encoded PNG image.
    
    Parameters
    ----------
    df              : DataFrame with lat, lon, avg_rad columns
    highlight_point : Optional dict with {lat, lon, color, label}
    
    Returns
    -------
    str : Base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="black")
    ax.set_facecolor("black")

    log_rad = np.log1p(df["avg_rad"])
    vmin = log_rad.quantile(0.02)
    vmax = log_rad.quantile(0.99)

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=log_rad, cmap=_VIIRS_CMAP,
        s=18, alpha=0.92, linewidths=0,
        vmin=vmin, vmax=vmax
    )

    if highlight_point:
        ax.scatter(
            [highlight_point["lon"]], [highlight_point["lat"]],
            c=highlight_point["color"],
            s=200, marker="*",
            edgecolors="white", linewidths=1.5, zorder=10
        )
        ax.annotate(
            f'  {highlight_point["label"]}',
            (highlight_point["lon"], highlight_point["lat"]),
            color="white", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Longitude", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Latitude",  color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Radiance log(nW/cm²/sr)", color="#aaaaaa", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#666666", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888888")

    title = "Heatmap · Nighttime Light Pollution\nCundinamarca – VIIRS/SUOMI-NPP"
    if highlight_point:
        title = f"Analyzed Point · {highlight_point['label']}\nCundinamarca – VIIRS/SUOMI-NPP"
    ax.set_title(title, color="white", fontsize=11, pad=12)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")



# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/")
def home():
    """Home page - project overview and navigation."""
    stats        = classifier.get_stats()
    zone_summary = classifier.get_zone_summary()
    heatmap_b64  = _render_heatmap(classifier.get_full_dataframe())

    return render_template(
        "home.html",
        stats=stats,
        zone_summary=zone_summary,
        heatmap_b64=heatmap_b64,
        zone_names=ZONE_NAMES,
        zone_colors=ZONE_COLORS
    )


@app.route("/methodology")
def methodology():
    """CRISP-ML(Q) Methodology explanation."""
    return render_template(
        "methodology.html",
        stats=classifier.get_stats(),
    )


@app.route("/business-understanding")
def business_understanding():
    """Phase 1: Business Understanding."""
    return render_template(
        "business_understanding.html",
        stats=classifier.get_stats(),
    )


@app.route("/data-understanding")
def data_understanding():
    """Phase 2: Data Understanding and exploratory analysis."""
    df   = classifier.get_full_dataframe()
    desc = df["avg_rad"].describe()

    descriptive_stats = {
        "Minimum":    round(float(desc["min"]),   4),
        "Median":     round(float(desc["50%"]),   4),
        "Mean":       round(float(desc["mean"]),  4),
        "Maximum":    round(float(desc["max"]),   4),
        "Std Dev":    round(float(desc["std"]),   4),
        "P95":        round(float(df["avg_rad"].quantile(.95)), 4),
    }

    return render_template(
        "data_understanding.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(df),
        descriptive_stats=descriptive_stats,
    )


@app.route("/data-engineering")
def data_engineering():
    """Phase 3: Data Engineering and preparation."""
    importances = classifier.clf.feature_importances_
    feat_data = [
        {"name": "avg_rad",     "pct": 0,  "desc": "Average VIIRS radiance"},
        {"name": "log_rad",     "pct": 0,  "desc": "log(1 + avg_rad) - reduces outlier bias"},
        {"name": "dist_bogota", "pct": 0,  "desc": "Distance to Bogotá centroid"},
        {"name": "rad_sq",      "pct": 0,  "desc": "avg_rad² - captures extreme values"},
        {"name": "lat",         "pct": 0,  "desc": "Latitude coordinate"},
        {"name": "lon",         "pct": 0,  "desc": "Longitude coordinate"},
    ]
    for i, feat in enumerate(feat_data):
        feat["pct"] = round(float(importances[i]) * 100, 1)
    feat_data.sort(key=lambda x: x["pct"], reverse=True)

    return render_template(
        "data_engineering.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        feature_importance=feat_data,
    )


@app.route("/model-training")
def model_training():
    """Phase 4: Model Training and Evaluation."""
    importances = classifier.clf.feature_importances_
    feat_data = [
        {"name": "avg_rad",     "pct": 0,  "desc": "Average VIIRS radiance"},
        {"name": "log_rad",     "pct": 0,  "desc": "log(1 + avg_rad)"},
        {"name": "dist_bogota", "pct": 0,  "desc": "Distance to Bogotá"},
        {"name": "rad_sq",      "pct": 0,  "desc": "Radiance squared"},
        {"name": "lat",         "pct": 0,  "desc": "Latitude"},
        {"name": "lon",         "pct": 0,  "desc": "Longitude"},
    ]
    for i, feat in enumerate(feat_data):
        feat["pct"] = round(float(importances[i]) * 100, 1)
    feat_data.sort(key=lambda x: x["pct"], reverse=True)

    model_info = {
        "accuracy": classifier.get_stats()["accuracy"],
        "params": {
            "algorithm":    "Gradient Boosting Classifier",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth":    5,
            "random_state": 42,
            "test_size":    "20%",
            "stratify":     "By pollution class",
        },
        "features": 6,
        "classes": 3,
        "training_samples": 6688,
        "test_samples": 1672,
    }

    return render_template(
        "model_training.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        model_info=model_info,
        feature_importance=feat_data,
        heatmap_b64=_render_heatmap(classifier.get_full_dataframe()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predict light pollution level for a given point."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON"}), 400

    try:
        lat     = float(data["lat"])
        lon     = float(data["lon"])
        avg_rad = float(data["avg_rad"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid parameters. Required: lat, lon, avg_rad"}), 400

    if not LightPollutionClassifier.validate_coords(lat, lon):
        b = GEO_BOUNDS
        return jsonify({
            "error": (
                f"Coordinates outside Cundinamarca. "
                f"Lat: {b['lat_min']}–{b['lat_max']}, "
                f"Lon: {b['lon_min']}–{b['lon_max']}"
            )
        }), 400

    if not (0 <= avg_rad <= 500):
        return jsonify({"error": "Radiance must be between 0 and 500 nW/cm²/sr"}), 400

    result = classifier.predict(lat, lon, avg_rad)

    heatmap_b64 = _render_heatmap(
        classifier.get_full_dataframe(),
        highlight_point={
            "lat":   lat,
            "lon":   lon,
            "color": result["zone_color"],
            "label": result["zone_name"]
        }
    )

    return jsonify({
        **result,
        "lat":         lat,
        "lon":         lon,
        "avg_rad":     avg_rad,
        "heatmap_b64": heatmap_b64
    })


@app.route("/api/data")
def api_data():
    """Return sample points for Leaflet map."""
    return jsonify(classifier.get_map_sample(n=500))


@app.route("/api/stats")
def api_stats():
    """Return model metrics and zone distribution."""
    stats = classifier.get_stats()
    return jsonify({
        "accuracy": stats["accuracy"],
        "total_points":    stats["total_points"],
        "model":    "Gradient Boosting Classifier",
        "features": LightPollutionClassifier.FEATURE_NAMES,
        "zones": {
            str(z): {
                "name":  ZONE_NAMES[z],
                "color": ZONE_COLORS[z],
                "count": int(stats["zone_counts"].get(z, 0))
            }
            for z in range(3)
        }
    })


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)

