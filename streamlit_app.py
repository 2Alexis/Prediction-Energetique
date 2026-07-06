"""
Prédiction de la demande énergétique quotidienne — application Streamlit.

Charge le meilleur modèle entraîné (voir src/train_model.py) et prédit la
consommation à partir des conditions météo / calendrier saisies. Les variables
autorégressives (lags) sont dérivées automatiquement de l'historique récent.

Lancement local :  streamlit run streamlit_app.py
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
SERIES_PATH = os.path.join(ROOT, "data", "processed", "daily_series.csv")

st.set_page_config(page_title="Prévision énergétique", page_icon="⚡", layout="wide")

MONTHS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
          "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


@st.cache_resource
def load_artifacts():
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "best_model_*.joblib")))
    model = joblib.load(model_files[-1]) if model_files else None

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), encoding="utf-8") as f:
        feature_cols = json.load(f)

    metrics_files = sorted(glob.glob(os.path.join(MODELS_DIR, "metrics_*.json")))
    metrics = {}
    if metrics_files:
        with open(metrics_files[-1], encoding="utf-8") as f:
            metrics = json.load(f)

    series = pd.read_csv(SERIES_PATH, parse_dates=["date"], index_col="date")["Consommation(MW)"]
    return model, feature_cols, metrics, series


model, FEATURES, METRICS, SERIES = load_artifacts()

# Meilleur modèle = plus haut R² test
best_name = max(METRICS, key=lambda n: METRICS[n]["test_metrics"]["r2"]) if METRICS else "random_forest"
best_r2 = METRICS[best_name]["test_metrics"]["r2"] if METRICS else 0.86
best_mae = METRICS[best_name]["test_metrics"]["mae"] if METRICS else 0

# ── En-tête ───────────────────────────────────────────────────────────────
st.title("⚡ Prévision de la demande énergétique")
st.caption(
    "Prédiction de la consommation électrique quotidienne d'une ville "
    "(météo + calendrier + historique). Modèle : "
    f"**{best_name.replace('_', ' ').title()}** · R² test **{best_r2:.2f}** · "
    f"erreur moyenne ± {best_mae:,.0f} MW."
)

tab_pred, tab_perf = st.tabs(["🔮 Prédiction", "📊 Performance du modèle"])

# ── Onglet Prédiction ─────────────────────────────────────────────────────
with tab_pred:
    left, right = st.columns([2, 1.4], gap="large")

    with left:
        st.subheader("Conditions")
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.slider("🌡️ Température (°C)", -10.0, 40.0, 15.0, 0.5)
            humidite = st.slider("💧 Humidité (%)", 0, 100, 60)
            vitesse_vent = st.slider("💨 Vent (km/h)", 0.0, 50.0, 10.0, 0.5)
        with c2:
            precipitation = st.slider("🌧️ Précipitations (mm)", 0.0, 50.0, 0.0, 0.5)
            mois = st.selectbox("📅 Mois", range(1, 13), format_func=lambda m: MONTHS[m - 1])
            jour_semaine = st.selectbox("🗓️ Jour", range(7), format_func=lambda d: DAYS[d])
        evenement = st.toggle("🎉 Événement spécial ce jour-là")
        predict = st.button("Prédire la consommation", type="primary", use_container_width=True)

    with right:
        st.subheader("Résultat")
        if predict and model is not None:
            weekend = 1 if jour_semaine >= 5 else 0
            import datetime as _dt
            jour_annee = _dt.date(2020, mois, 15).timetuple().tm_yday
            # Lags dérivés de l'historique le plus récent
            lag1 = float(SERIES.iloc[-1])
            lag7 = float(SERIES.iloc[-7])
            moyenne_7j = float(SERIES.iloc[-7:].mean())
            moyenne_30j = float(SERIES.iloc[-30:].mean())
            values = {
                "temperature": temperature, "humidite": humidite,
                "vitesse_vent": vitesse_vent, "precipitation": precipitation,
                "evenement": int(evenement), "mois": mois,
                "jour_semaine": jour_semaine, "weekend": weekend,
                "jour_annee": jour_annee, "lag1": lag1, "lag7": lag7,
                "moyenne_7j": moyenne_7j, "moyenne_30j": moyenne_30j,
            }
            X = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
            pred = float(model.predict(X)[0])
            st.metric("Consommation prédite", f"{pred:,.0f} MW")
            delta = (pred - lag1) / lag1 * 100
            st.metric("Vs. dernier jour connu", f"{lag1:,.0f} MW", f"{delta:+.1f}%")
        else:
            st.info("Règle les conditions à gauche puis clique sur **Prédire**.")

# ── Onglet Performance ────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Comparaison des modèles (R² sur test)")
    if METRICS:
        rows = [
            {"Modèle": n.replace("_", " ").title(),
             "R² test": round(m["test_metrics"]["r2"], 3),
             "RMSE": round(m["test_metrics"]["rmse"]),
             "MAE": round(m["test_metrics"]["mae"])}
            for n, m in sorted(METRICS.items(), key=lambda kv: -kv[1]["test_metrics"]["r2"])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "🔑 La consommation étant fortement autocorrélée (0,90 avec la veille), "
        "l'ajout de variables autorégressives (lags) fait passer le R² test de "
        "~0,53 (météo seule) à **0,86**."
    )

    col1, col2 = st.columns(2)
    for col, img, cap in [
        (col1, "model_comparison.png", "Comparaison des modèles"),
        (col2, "feature_importance.png", "Importance des variables"),
    ]:
        path = os.path.join(RESULTS_DIR, img)
        if os.path.exists(path):
            col.image(path, caption=cap, use_container_width=True)

    pva = os.path.join(RESULTS_DIR, "predictions_vs_actual.png")
    if os.path.exists(pva):
        st.image(pva, caption="Consommation réelle vs prédite (période de test)", use_container_width=True)

st.divider()
st.caption("Alexis Clerc · [GitHub](https://github.com/2Alexis) · [Portfolio](https://alexis-clerc.fr)")
