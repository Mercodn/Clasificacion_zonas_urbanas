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
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

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

    title = "Heatmap · Light Pollution\nCundinamarca – VIIRS/SUOMI-NPP"
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


def _render_roc_curve(classifier):
    """Genera un ROC plot multi-clase en base64."""
    if not classifier._trained:
        return ""

    y_test = classifier.y_test
    y_proba = classifier.y_proba_test
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="black")
    ax.set_facecolor("black")

    colors = ["#38bef8", "#22c55e", "#ffb300", "#ff1100", "#00cfff", "#0040ff"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            label=f"{ZONE_NAMES[i]} (AUC {roc_auc:.2f})",
            color=colors[i % len(colors)], linewidth=2
        )

    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.5)
    ax.set_xlabel("False Positive Rate", color="#cccccc")
    ax.set_ylabel("True Positive Rate", color="#cccccc")
    ax.set_title("Multi-class ROC Curve — Final Model", color="white", fontsize=11, pad=12)
    ax.tick_params(colors="#999999")
    ax.legend(loc="lower right", fontsize=8, facecolor="#111111", framealpha=0.9, edgecolor="#444444")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _render_confusion_matrix(classifier):
    """Generate a confusion matrix plot and return it as base64."""
    if not classifier._trained:
        return ""

    y_true = classifier.y_test
    y_pred = np.argmax(classifier.y_proba_test, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(ZONE_NAMES)))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(cm, cmap="viridis")

    ax.set_title("Confusion Matrix — Test Set", color="white", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Zone", color="#cccccc")
    ax.set_ylabel("Actual Zone", color="#cccccc")
    ax.set_xticks(np.arange(len(ZONE_NAMES)))
    ax.set_yticks(np.arange(len(ZONE_NAMES)))
    ax.set_xticklabels([ZONE_NAMES[i] for i in range(len(ZONE_NAMES))], rotation=45, ha="right", color="#cccccc", fontsize=8)
    ax.set_yticklabels([ZONE_NAMES[i] for i in range(len(ZONE_NAMES))], color="#cccccc", fontsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")
    ax.tick_params(colors="#888888")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def business_understanding():
    return render_template(
        "business_understanding.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
    )


def data_understanding_page():
    df   = classifier.get_full_dataframe()
    desc = df["avg_rad"].describe()

    descriptive_stats = {
        "Mínimo":  round(float(desc["min"]), 4),
        "Mediana": round(float(desc["50%"]), 4),
        "Media":   round(float(desc["mean"]), 4),
        "Máximo":  round(float(desc["max"]), 4),
        "Std Dev": round(float(desc["std"]), 4),
        "P95":     round(float(df["avg_rad"].quantile(.95)), 4),
    }

    return render_template(
        "data_understanding.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(df),
        descriptive_stats=descriptive_stats,
    )


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
        "model":    stats["model_name"],
        "best_model": stats["model_name"],
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
    return data_understanding_page()


@app.route("/fase2")
def fase2():
    """CRISP-DM Fase 2 · Modelado ML"""
    feat_data = classifier.get_feature_importance()

    best_model = classifier.get_best_model_info()
    model_info = {
        "accuracy": classifier.get_stats()["accuracy"],
        "name":     classifier.best_model_name,
        "description": best_model.get("description", ""),
        "params":  best_model.get("params", {}),
        "methods": {
            "train()":             "Carga CSV, etiqueta zonas, entrena los modelos comparativos",
            "predict()":           "Clasifica un punto → zona + confianza + probabilidades con el mejor modelo",
            "get_stats()":         "Retorna métricas globales del dataset y el modelo seleccionado",
            "get_zone_summary()":  "Lista de zonas con conteo y porcentaje",
            "get_map_sample()":    "Muestra de 500 puntos para el mapa Leaflet",
            "get_model_comparison()": "Retorna la comparación entre los 3 modelos entrenados",
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
    return business_understanding()


@app.route("/data-understanding")
def data_understanding():
    return data_understanding_page()


@app.route("/data-engineering")
def data_engineering():
    return fase2()


@app.route("/model-engineering")
@app.route("/model-development")
def model_engineering():
    return render_template(
        "model_engineering.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        model_reports=classifier.get_model_comparison(),
        best_model=classifier.get_best_model_info(),
        best_model_name=classifier.best_model_name,
        heatmap_b64=_render_heatmap(classifier.get_full_dataframe()),
        roc_curve_b64=_render_roc_curve(classifier),
        feature_importance=classifier.get_feature_importance(),
    )


@app.route("/evaluation")
def evaluation():
    return fase3()


@app.route("/fase3")
def fase3():
    """CRISP-DM Phase 3 · Evaluation & Deployment"""
    return render_template(
        "fase3_evaluacion.html",
        stats=classifier.get_stats(),
        zone_summary=classifier.get_zone_summary(),
        heatmap_b64=_render_heatmap(classifier.get_full_dataframe()),
        roc_curve_b64=_render_roc_curve(classifier),
        confusion_matrix_b64=_render_confusion_matrix(classifier),
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
