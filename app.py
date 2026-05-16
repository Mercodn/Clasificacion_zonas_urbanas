# ══════════════════════════════════════════════════════════════
#  app.py  ·  Light Pollution Classifier
#  Servidor Flask — lógica de rutas y visualización
#  JoanMoreno · Cundinamarca VIIRS / SUOMI-NPP
# ══════════════════════════════════════════════════════════════

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
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
