"""
Entraînement du modèle de prévision de la demande énergétique quotidienne.

Pipeline :
  1. Reconstruction d'une série journalière propre à partir des données brutes.
  2. Feature engineering : météo + calendrier + variables autorégressives (lags, moyennes glissantes).
  3. Découpage TEMPOREL (80% passé / 20% futur) pour une évaluation réaliste, sans fuite.
  4. Comparaison de 4 modèles (RandomForest, GradientBoosting, XGBoost, LightGBM).
  5. Sauvegarde du meilleur modèle, des métriques, de l'importance des variables et des visualisations.

Usage : python src/train_model.py
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "cleaned_file.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

FEATURES = [
    "temperature", "humidite", "vitesse_vent", "precipitation", "evenement",
    "mois", "jour_semaine", "weekend", "jour_annee",
    "lag1", "lag7", "moyenne_7j", "moyenne_30j",
]


def build_daily_series():
    """Reconstruit une série journalière régulière à partir des données brutes."""
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["Date_Debut_x"])
    daily = (
        df.groupby("date")
        .agg({
            "Consommation(MW)": "mean",
            "Temperature": "mean",
            "Humidity": "mean",
            "Wind Speed": "mean",
            "Precipitation": "mean",
            "Event": "max",
        })
        .sort_index()
        .asfreq("D")
    )
    daily["Consommation(MW)"] = daily["Consommation(MW)"].interpolate()
    daily = daily.ffill().bfill()
    return daily


def build_features(daily):
    y = daily["Consommation(MW)"]
    f = pd.DataFrame(index=daily.index)
    f["temperature"] = daily["Temperature"]
    f["humidite"] = daily["Humidity"]
    f["vitesse_vent"] = daily["Wind Speed"]
    f["precipitation"] = daily["Precipitation"]
    f["evenement"] = daily["Event"]
    f["mois"] = daily.index.month
    f["jour_semaine"] = daily.index.dayofweek
    f["weekend"] = (daily.index.dayofweek >= 5).astype(int)
    f["jour_annee"] = daily.index.dayofyear
    # Variables autorégressives (décalées d'au moins 1 jour -> aucune fuite)
    f["lag1"] = y.shift(1)
    f["lag7"] = y.shift(7)
    f["moyenne_7j"] = y.shift(1).rolling(7).mean()
    f["moyenne_30j"] = y.shift(1).rolling(30).mean()
    f["consommation"] = y
    return f.dropna()


def metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    daily = build_daily_series()
    daily.to_csv(os.path.join(PROCESSED_DIR, "daily_series.csv"))
    data = build_features(daily)

    X, y = data[FEATURES], data["consommation"]
    cut = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    print(f"Série journalière : {len(daily)} jours | train {len(X_train)} / test {len(X_test)}")
    print(f"Période test : {X_test.index.min().date()} -> {X_test.index.max().date()}\n")

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1
        ),
    }

    all_metrics, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        all_metrics[name] = {
            "train_metrics": metrics(y_train, model.predict(X_train)),
            "test_metrics": metrics(y_test, model.predict(X_test)),
        }
        m = all_metrics[name]["test_metrics"]
        print(f"{name:18} test R²={m['r2']:.3f} | RMSE={m['rmse']:,.0f} | MAE={m['mae']:,.0f}")

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["test_metrics"]["r2"])
    best_model = fitted[best_name]
    print(f"\n>>> Meilleur modèle : {best_name} (R² test = {all_metrics[best_name]['test_metrics']['r2']:.3f})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"best_model_{ts}.joblib"))
    with open(os.path.join(MODELS_DIR, f"metrics_{ts}.json"), "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)  # format plat : {modele: {train_metrics, test_metrics}}
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w", encoding="utf-8") as fh:
        json.dump(FEATURES, fh, indent=2)

    # Importance des variables
    if hasattr(best_model, "feature_importances_"):
        imp = (
            pd.Series(best_model.feature_importances_, index=FEATURES)
            .sort_values(ascending=False)
        )
        imp.rename("importance").to_csv(os.path.join(MODELS_DIR, f"feature_importance_{ts}.csv"))
        plt.figure(figsize=(9, 5))
        imp.iloc[::-1].plot.barh(color="#764ba2")
        plt.title("Importance des variables — " + best_name)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=120)
        plt.close()

    # Prédictions vs réel (période de test)
    y_pred = best_model.predict(X_test)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label="Réel", color="#8892b0", lw=1.5)
    plt.plot(y_test.index, y_pred, label="Prédit", color="#764ba2", lw=1.5)
    plt.title(f"Consommation — réel vs prédit (test) | {best_name} · R²={all_metrics[best_name]['test_metrics']['r2']:.3f}")
    plt.ylabel("Consommation (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictions_vs_actual.png"), dpi=120)
    plt.close()

    # Comparaison des modèles
    names = list(all_metrics.keys())
    r2s = [all_metrics[n]["test_metrics"]["r2"] for n in names]
    plt.figure(figsize=(8, 4.5))
    plt.bar(names, r2s, color="#764ba2")
    plt.ylabel("R² (test)")
    plt.title("Comparaison des modèles")
    plt.ylim(0, 1)
    for i, v in enumerate(r2s):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    print("\nArtefacts sauvegardés : models/, results/ (comparaison, importances, prédictions).")


if __name__ == "__main__":
    main()
