# ══════════════════════════════════════════════════════════════
#  model.py  ·  Light Pollution Classifier
#  Módulo de Machine Learning — separado de la lógica Flask
#  JoanMoreno · Cundinamarca VIIRS / SUOMI-NPP
# ══════════════════════════════════════════════════════════════

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────
# CONSTANTES DE DOMINIO
# ─────────────────────────────────────────

ZONE_NAMES = {
    0: "Pristine Natural",
    1: "Rural Low",
    2: "Suburban",
    3: "Urban Moderate",
    4: "Urban High",
    5: "Metropolitan"
}

ZONE_COLORS = {
    0: "#000080",
    1: "#0040FF",
    2: "#00CFFF",
    3: "#00FF80",
    4: "#FFB300",
    5: "#FF1100"
}

ZONE_DESC = {
    0: "Nearly pristine sky. Minimal light interference. Ideal for astronomy.",
    1: "Rural area with low density. Very light pollution.",
    2: "Urban periphery. City glow visible on the horizon.",
    3: "Moderate residential or industrial area. Limited visible stars.",
    4: "Dense urban center. Orange sky. High light pollution.",
    5: "Metropolitan core. Extreme radiance. Completely bright sky."
}

# Coordenadas de referencia — centroide Bogotá
BOGOTA_LAT = 4.711
BOGOTA_LON = -74.0721

# Límites geográficos de Cundinamarca
GEO_BOUNDS = {
    "lat_min": 3.6, "lat_max": 5.9,
    "lon_min": -75.0, "lon_max": -72.9
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "data_prepared.csv")


# ─────────────────────────────────────────
# FUNCIONES DE PREPROCESAMIENTO
# ─────────────────────────────────────────

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga y parsea el CSV de VIIRS con coordenadas GeoJSON."""
    df = pd.read_csv(path)
    df["lon"] = df[".geo"].apply(lambda x: json.loads(x)["coordinates"][0])
    df["lat"] = df[".geo"].apply(lambda x: json.loads(x)["coordinates"][1])
    df["avg_rad"] = pd.to_numeric(df["avg_rad"], errors="coerce")
    df = df.dropna(subset=["avg_rad"]).reset_index(drop=True)
    return df


def label_zone(avg_rad: float) -> int:
    """
    Clasifica un punto según su radiancia promedio VIIRS (nW/cm²/sr).
    Umbrales basados en la escala de Bortle adaptada a datos satelitales.
    """
    if avg_rad < 0.20:
        return 0   # Oscuro Natural
    elif avg_rad < 0.40:
        return 1   # Rural Bajo
    elif avg_rad < 1.00:
        return 2   # Suburbano
    elif avg_rad < 5.00:
        return 3   # Urbano Moderado
    elif avg_rad < 20.0:
        return 4   # Urbano Alto
    else:
        return 5   # Metropolitano


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingeniería de features a partir del dataset VIIRS.
    Retorna un DataFrame listo para entrenar o predecir.
    """
    feats = pd.DataFrame()
    feats["avg_rad"]      = df["avg_rad"]
    feats["log_rad"]      = np.log1p(df["avg_rad"])
    feats["lat"]          = df["lat"]
    feats["lon"]          = df["lon"]
    feats["dist_bogota"]  = np.sqrt(
        (df["lat"] - BOGOTA_LAT) ** 2 + (df["lon"] - BOGOTA_LON) ** 2
    )
    feats["rad_sq"]       = df["avg_rad"] ** 2
    return feats


def build_single_feature(lat: float, lon: float, avg_rad: float) -> pd.DataFrame:
    """Construye el vector de features para un único punto de predicción."""
    return pd.DataFrame([{
        "avg_rad":     avg_rad,
        "log_rad":     np.log1p(avg_rad),
        "lat":         lat,
        "lon":         lon,
        "dist_bogota": np.sqrt((lat - BOGOTA_LAT)**2 + (lon - BOGOTA_LON)**2),
        "rad_sq":      avg_rad ** 2
    }])


# ─────────────────────────────────────────
# CLASE PRINCIPAL DEL MODELO
# ─────────────────────────────────────────

class LightPollutionClassifier:
    """
    Clasificador de contaminación lumínica basado en datos VIIRS.
    Encapsula entrenamiento, comparación de modelos, predicción y métricas.

    Esta implementación entrena tres algoritmos distintos y elige el mejor
    según la métrica de accuracy en el conjunto de prueba.
    """

    FEATURE_NAMES = ["avg_rad", "log_rad", "lat", "lon", "dist_bogota", "rad_sq"]

    MODEL_LIBRARY = {
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "Logistic Regression": lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=1200,
            random_state=42
        ),
    }

    MODEL_DESCRIPTIONS = {
        "Gradient Boosting": "Boosted trees that capture non-linear radial and geographic patterns.",
        "Random Forest": "Bagged tree ensemble that increases robustness and reduces variance.",
        "Logistic Regression": "Baseline linear model for multi-class classification with calibrated probabilities.",
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {name: builder() for name, builder in self.MODEL_LIBRARY.items()}
        self.fitted_models = {}
        self.model_results = []
        self.best_model_name = None
        self.clf = None
        self.accuracy = 0.0
        self.df = None          # dataset completo con predicciones
        self._trained = False
        self.y_test = None      # para ROC curve
        self.y_proba_test = None
        self.y_pred = None

    # ── ENTRENAMIENTO ──────────────────────

    def train(self, data_path: str = DATA_PATH) -> float:
        """
        Carga el dataset, entrena los tres modelos y elige el mejor.
        Retorna la mejor exactitud sobre el conjunto de prueba (20%).
        """
        # 1. Cargar y etiquetar
        df = load_dataset(data_path)
        df["zone"] = df["avg_rad"].apply(label_zone)

        # 2. Features y escalado
        X = build_features(df)
        y = df["zone"]

        # 3. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_scaled = self.scaler.transform(X)

        # 4. Entrenar cada modelo y registrar métricas
        self.fitted_models = {}
        self.model_results = []
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            self.fitted_models[name] = model
            self.model_results.append({
                "name": name,
                "accuracy": round(accuracy * 100, 1),
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1_score": round(f1 * 100, 1),
                "description": self.MODEL_DESCRIPTIONS.get(name, ""),
                "params": model.get_params(),
            })

        # 5. Seleccionar el mejor modelo
        self.best_model_name = max(self.model_results, key=lambda r: r["accuracy"])["name"]
        self.clf = self.fitted_models[self.best_model_name]
        self.y_test = y_test
        self.y_pred = self.clf.predict(X_test_scaled)
        self.y_proba_test = self.clf.predict_proba(X_test_scaled)
        self.accuracy = float(accuracy_score(y_test, self.y_pred))

        # 6. Predicción sobre todo el dataset con el mejor modelo
        df["pred_zone"] = self.clf.predict(X_scaled)
        df["pred_name"] = df["pred_zone"].map(ZONE_NAMES)
        df["pred_color"] = df["pred_zone"].map(ZONE_COLORS)

        self.df = df
        self._trained = True

        print(f"[LightPollutionClassifier] Trained best model: {self.best_model_name} · Accuracy: {self.accuracy:.4f}")
        return self.accuracy

    # ── PREDICCIÓN ─────────────────────────

    def predict(self, lat: float, lon: float, avg_rad: float) -> dict:
        """
        Predice la zona de contaminación lumínica para un punto geográfico.

        Parámetros
        ----------
        lat     : latitud decimal
        lon     : longitud decimal
        avg_rad : radiancia promedio VIIRS (nW/cm²/sr)

        Retorna
        -------
        dict con zone_id, zone_name, zone_color, zone_desc,
             confidence, probabilities
        """
        if not self._trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")

        feats        = build_single_feature(lat, lon, avg_rad)
        feats_scaled = self.scaler.transform(feats)

        zone_id = int(self.clf.predict(feats_scaled)[0])
        proba   = self.clf.predict_proba(feats_scaled)[0]

        return {
            "zone_id":       zone_id,
            "zone_name":     ZONE_NAMES[zone_id],
            "zone_color":    ZONE_COLORS[zone_id],
            "zone_desc":     ZONE_DESC[zone_id],
            "confidence":    round(float(proba[zone_id]) * 100, 1),
            "probabilities": {
                ZONE_NAMES[i]: round(float(p) * 100, 1)
                for i, p in enumerate(proba)
            }
        }

    # ── MÉTRICAS Y ACCESO A DATOS ──────────

    def get_stats(self) -> dict:
        """Retorna estadísticas generales del dataset y el modelo."""
        self._check_trained()
        zone_counts = self.df["pred_zone"].value_counts().to_dict()
        return {
            "total_points": len(self.df),
            "accuracy":     round(self.accuracy * 100, 1),
            "model_name":   self.best_model_name,
            "max_rad":      round(float(self.df["avg_rad"].max()), 3),
            "mean_rad":     round(float(self.df["avg_rad"].mean()), 3),
            "zone_counts":  zone_counts
        }

    def get_model_comparison(self) -> list:
        """Retorna las métricas comparativas de los modelos entrenados."""
        self._check_trained()
        return self.model_results

    def get_best_model_info(self) -> dict:
        """Retorna información del modelo seleccionado para producción."""
        self._check_trained()
        results = {item["name"]: item for item in self.model_results}
        return results.get(self.best_model_name, {})

    def get_feature_importance(self) -> list:
        """Retorna importancias de feature para el modelo seleccionado."""
        self._check_trained()
        model = self.clf
        values = None
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.mean(np.abs(model.coef_), axis=0)
        else:
            values = np.zeros(len(self.FEATURE_NAMES))

        importance = [
            {"name": name, "value": float(values[i])}
            for i, name in enumerate(self.FEATURE_NAMES)
        ]
        importance.sort(key=lambda x: x["value"], reverse=True)
        total = sum(item["value"] for item in importance) or 1.0
        for item in importance:
            item["pct"] = round(item["value"] / total * 100, 1)
            item["desc"] = ""  # Descriptions can be added at template level if needed.
        return importance

    def get_zone_summary(self) -> list:
        """Retorna lista de zonas con conteo y porcentaje."""
        self._check_trained()
        total = len(self.df)
        summary = []
        for z in range(6):
            cnt = int((self.df["pred_zone"] == z).sum())
            summary.append({
                "id":    z,
                "name":  ZONE_NAMES[z],
                "color": ZONE_COLORS[z],
                "count": cnt,
                "pct":   round(cnt / total * 100, 1)
            })
        return summary

    def get_map_sample(self, n: int = 500) -> list:
        """Retorna una muestra aleatoria del dataset para el mapa Leaflet."""
        self._check_trained()
        cols   = ["lat", "lon", "avg_rad", "pred_zone", "pred_name", "pred_color"]
        sample = self.df[cols].sample(n=min(n, len(self.df)), random_state=1)
        return sample.to_dict(orient="records")

    def get_full_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame completo con predicciones."""
        self._check_trained()
        return self.df

    # ── HELPERS ────────────────────────────

    def _check_trained(self):
        if not self._trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")

    @staticmethod
    def validate_coords(lat: float, lon: float) -> bool:
        """Verifica que las coordenadas estén dentro de Cundinamarca."""
        b = GEO_BOUNDS
        return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]
