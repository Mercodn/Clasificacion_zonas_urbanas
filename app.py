# ══════════════════════════════════════════════════════════════
#  app.py  ·  Light Pollution Classifier
#  Servidor Flask — lógica de rutas y visualización
#  JoanMoreno · Cundinamarca VIIRS / SUOMI-NPP
# ══════════════════════════════════════════════════════════════

import os
import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from flask import Flask, render_template, request, jsonify

# ── Importar el módulo de ML ──────────────────────────────────
from model import (
    LightPollutionClassifier,
    ZONE_NAMES,
    ZONE_COLORS,
    GEO_BOUNDS,
    DATA_PATH,
)

# ─────────────────────────────────────────
# INICIALIZAR FLASK
# ─────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────
# ENTRENAR MODELO AL ARRANCAR
# ─────────────────────────────────────────
classifier = LightPollutionClassifier()
classifier.train(DATA_PATH)


# ─────────────────────────────────────────
# VISUALIZACIÓN · MAPA DE CALOR
# ─────────────────────────────────────────

_VIIRS_COLORS = [
    "#000000", "#000033", "#000080", "#0040FF",
    "#00CFFF", "#00FF80", "#FFFF00", "#FFB300", "#FF4400", "#FF0000"
]
_VIIRS_CMAP = LinearSegmentedColormap.from_list("viirs", _VIIRS_COLORS, N=512)


def _render_heatmap(df, highlight_point: dict = None) -> str:
    """
    Genera el mapa de calor VIIRS como imagen PNG en base64.

    Parámetros
    ----------
    df              : DataFrame con columnas lat, lon, avg_rad
    highlight_point : dict opcional con {lat, lon, color, label}
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

    ax.set_xlabel("Longitud", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Latitud",  color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Radiancia log(nW/cm²/sr)", color="#aaaaaa", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#666666", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888888")

    title = "Mapa de Calor · Contaminación Lumínica\nCundinamarca – VIIRS/SUOMI-NPP"
    if highlight_point:
        title = f"Punto Analizado · {highlight_point['label']}\nCundinamarca – VIIRS/SUOMI-NPP"
    ax.set_title(title, color="white", fontsize=11, pad=12)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────

@app.route("/")
def index():
    stats        = classifier.get_stats()
    zone_summary = classifier.get_zone_summary()
    heatmap_b64  = _render_heatmap(classifier.get_full_dataframe())

    return render_template(
        "index.html",
        stats=stats,
        zone_summary=zone_summary,
        heatmap_b64=heatmap_b64,
        zone_names=ZONE_NAMES,
        zone_colors=ZONE_COLORS
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    try:
        lat     = float(data["lat"])
        lon     = float(data["lon"])
        avg_rad = float(data["avg_rad"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Parámetros inválidos. Se requieren: lat, lon, avg_rad"}), 400

    if not LightPollutionClassifier.validate_coords(lat, lon):
        b = GEO_BOUNDS
        return jsonify({
            "error": (
                f"Coordenadas fuera de Cundinamarca. "
                f"Lat: {b['lat_min']}–{b['lat_max']}, "
                f"Lon: {b['lon_min']}–{b['lon_max']}"
            )
        }), 400

    if not (0 <= avg_rad <= 500):
        return jsonify({"error": "La radiancia debe estar entre 0 y 500 nW/cm²/sr"}), 400

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
    """Retorna muestra de puntos para el mapa Leaflet."""
    return jsonify(classifier.get_map_sample(n=500))


@app.route("/api/stats")
def api_stats():
    """Retorna métricas del modelo y distribución de zonas."""
    stats = classifier.get_stats()
    return jsonify({
        "accuracy": stats["accuracy"],
        "total":    stats["total_points"],
        "model":    "Gradient Boosting Classifier",
        "features": LightPollutionClassifier.FEATURE_NAMES,
        "zones": {
            str(z): {
                "name":  ZONE_NAMES[z],
                "color": ZONE_COLORS[z],
                "count": int(stats["zone_counts"].get(z, 0))
            }
            for z in range(6)
        }
    })


# ─────────────────────────────────────────
# RUTAS CRISP-DM
# ─────────────────────────────────────────

@app.route("/fase1")
def fase1():
    """CRISP-DM Fase 1 · Entendimiento de Datos"""
    df   = classifier.get_full_dataframe()
    desc = df["avg_rad"].describe()

    descriptive_stats = {
        "Mínimo":    round(float(desc["min"]),   4),
        "Mediana":   round(float(desc["50%"]),   4),
        "Media":     round(float(desc["mean"]),  4),
        "Máximo":    round(float(desc["max"]),   4),
        "Std Dev":   round(float(desc["std"]),   4),
        "P95":       round(float(df["avg_rad"].quantile(.95)), 4),
    }

    return render_template(
        "fase1_datos.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(df),
        descriptive_stats=descriptive_stats,
    )


@app.route("/fase2")
def fase2():
    """CRISP-DM Fase 2 · Modelado ML"""
    importances = classifier.clf.feature_importances_
    feat_data = [
        {"name": "avg_rad",     "pct": 0,  "desc": "Radiancia promedio VIIRS bruta"},
        {"name": "log_rad",     "pct": 0,  "desc": "log(1 + avg_rad) · reduce sesgo outliers"},
        {"name": "dist_bogota", "pct": 0,  "desc": "Distancia al centroide de Bogotá"},
        {"name": "rad_sq",      "pct": 0,  "desc": "Radiancia al cuadrado · captura extremos"},
        {"name": "lat",         "pct": 0,  "desc": "Latitud decimal del punto"},
        {"name": "lon",         "pct": 0,  "desc": "Longitud decimal del punto"},
    ]
    for i, feat in enumerate(feat_data):
        feat["pct"] = round(float(importances[i]) * 100, 1)
    feat_data.sort(key=lambda x: x["pct"], reverse=True)

    model_info = {
        "accuracy": classifier.get_stats()["accuracy"],
        "params": {
            "n_estimators":  200,
            "learning_rate": 0.1,
            "max_depth":     5,
            "random_state":  42,
            "test_size":     "20%",
            "stratify":      "True",
        },
        "methods": {
            "train()":          "Carga CSV, etiqueta zonas, entrena el modelo",
            "predict()":        "Clasifica un punto → zona + confianza + probabilidades",
            "get_stats()":      "Retorna métricas globales del dataset y el modelo",
            "get_zone_summary()": "Lista de zonas con conteo y porcentaje",
            "get_map_sample()": "Muestra de 500 puntos para el mapa Leaflet",
        },
    }

    return render_template(
        "fase2_modelo.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        model_info=model_info,
        feature_importance=feat_data,
    )


@app.route("/methodology")
def methodology():
    """CRISP-ML methodology overview."""
    return render_template(
        "methodology.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(classifier.get_full_dataframe()),
    )


@app.route("/business")
def business():
    return fase1()


@app.route("/data-understanding")
def data_understanding():
    return fase1()


@app.route("/data-engineering")
def data_engineering():
    return fase2()


@app.route("/fase3")
def fase3():
    """CRISP-DM Phase 3 · Evaluation & Deployment"""
    return render_template(
        "fase3_evaluacion.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(classifier.get_full_dataframe()),
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
