"""
Prédiction de la demande énergétique quotidienne — application Streamlit.

Le modèle (Random Forest) est entraîné au démarrage puis mis en cache : pas de
fichier modèle à charger, aucune dépendance lourde au runtime. La méthodologie
complète (comparaison XGBoost/LightGBM, tuning) est dans src/train_model.py.

Lancement local :  streamlit run streamlit_app.py
"""

import os
import glob
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw", "cleaned_file.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

st.set_page_config(page_title="Prévision énergétique", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp {
    background:
      radial-gradient(1100px 550px at 15% -10%, rgba(255,184,0,.10), transparent 60%),
      radial-gradient(900px 480px at 90% 5%, rgba(0,229,255,.10), transparent 55%),
      linear-gradient(160deg,#0a0e27 0%,#0b1020 55%,#081018 100%);
}
#MainMenu, header, footer {visibility:hidden;}
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.hero {text-align:center; padding:1.2rem 1rem .3rem;}
.hero .badge {display:inline-block; font-size:.72rem; letter-spacing:2px; text-transform:uppercase;
    color:#FFB800; border:1px solid rgba(255,184,0,.35); border-radius:999px; padding:.3rem .95rem;
    margin-bottom:.9rem; background:rgba(255,184,0,.06);}
.hero h1 {font-size:2.9rem; font-weight:800; margin:.1rem 0;
    background:linear-gradient(110deg,#FFD166,#FF6B35 45%,#00E5FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#93a4c9; font-size:1.02rem; max-width:720px; margin:.35rem auto 0;}
.stButton > button {background:linear-gradient(120deg,#FFB800,#FF6B35)!important; color:#0a0e27!important;
    font-weight:700!important; border:none!important; border-radius:12px!important;}
.stButton > button:hover {filter:brightness(1.08); box-shadow:0 6px 22px rgba(255,140,0,.35);}
[data-testid="stMetric"] {background:rgba(255,255,255,.03); border:1px solid rgba(255,184,0,.18);
    border-radius:16px; padding:1rem 1.1rem;}
.stTabs [aria-selected="true"] {color:#FFB800 !important;}
hr {border-color:rgba(255,184,0,.15);}
</style>
""", unsafe_allow_html=True)

MONTHS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
          "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

FEATURES = ["temperature", "humidite", "vitesse_vent", "precipitation", "evenement",
            "mois", "jour_semaine", "weekend", "jour_annee",
            "lag1", "lag7", "moyenne_7j", "moyenne_30j"]


@st.cache_resource
def train():
    """Reconstruit la série journalière, entraîne le Random Forest, met en cache."""
    df = pd.read_csv(RAW)
    df["date"] = pd.to_datetime(df["Date_Debut_x"])
    daily = (df.groupby("date").agg({
        "Consommation(MW)": "mean", "Temperature": "mean", "Humidity": "mean",
        "Wind Speed": "mean", "Precipitation": "mean", "Event": "max"})
        .sort_index().asfreq("D"))
    daily["Consommation(MW)"] = daily["Consommation(MW)"].interpolate()
    daily = daily.ffill().bfill()
    y = daily["Consommation(MW)"]

    f = pd.DataFrame(index=daily.index)
    f["temperature"] = daily["Temperature"]; f["humidite"] = daily["Humidity"]
    f["vitesse_vent"] = daily["Wind Speed"]; f["precipitation"] = daily["Precipitation"]
    f["evenement"] = daily["Event"]; f["mois"] = daily.index.month
    f["jour_semaine"] = daily.index.dayofweek; f["weekend"] = (daily.index.dayofweek >= 5).astype(int)
    f["jour_annee"] = daily.index.dayofyear
    f["lag1"] = y.shift(1); f["lag7"] = y.shift(7)
    f["moyenne_7j"] = y.shift(1).rolling(7).mean(); f["moyenne_30j"] = y.shift(1).rolling(30).mean()
    f["y"] = y
    f = f.dropna()

    X, target = f[FEATURES], f["y"]
    cut = int(len(f) * 0.8)
    model = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=4,
                                  random_state=42, n_jobs=-1)
    model.fit(X.iloc[:cut], target.iloc[:cut])
    r2 = r2_score(target.iloc[cut:], model.predict(X.iloc[cut:]))
    mae = mean_absolute_error(target.iloc[cut:], model.predict(X.iloc[cut:]))
    return model, y, r2, mae


@st.cache_data
def load_metrics():
    files = sorted(glob.glob(os.path.join(ROOT, "models", "metrics_*.json")))
    if files:
        with open(files[-1], encoding="utf-8") as fh:
            return json.load(fh)
    return {}


model, SERIES, R2, MAE = train()
METRICS = load_metrics()

st.markdown(f"""
<div class="hero">
    <span class="badge">⚡ Machine Learning · Séries temporelles</span>
    <h1>Prévision de la demande énergétique</h1>
    <p>Consommation électrique quotidienne prédite à partir de la météo, du calendrier
    et de l'historique. Modèle Random Forest · R² test {R2:.2f} · erreur moyenne ± {MAE:,.0f} MW.</p>
</div>
""", unsafe_allow_html=True)

tab_pred, tab_perf = st.tabs(["🔮 Prédiction", "📊 Performance du modèle"])

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
        if predict:
            weekend = 1 if jour_semaine >= 5 else 0
            import datetime as _dt
            jour_annee = _dt.date(2020, mois, 15).timetuple().tm_yday
            lag1 = float(SERIES.iloc[-1]); lag7 = float(SERIES.iloc[-7])
            moyenne_7j = float(SERIES.iloc[-7:].mean()); moyenne_30j = float(SERIES.iloc[-30:].mean())
            values = {"temperature": temperature, "humidite": humidite,
                      "vitesse_vent": vitesse_vent, "precipitation": precipitation,
                      "evenement": int(evenement), "mois": mois, "jour_semaine": jour_semaine,
                      "weekend": weekend, "jour_annee": jour_annee, "lag1": lag1, "lag7": lag7,
                      "moyenne_7j": moyenne_7j, "moyenne_30j": moyenne_30j}
            X = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
            pred = float(model.predict(X)[0])
            st.metric("Consommation prédite", f"{pred:,.0f} MW")
            st.metric("Vs. dernier jour connu", f"{lag1:,.0f} MW", f"{(pred - lag1) / lag1 * 100:+.1f}%")
        else:
            st.info("Règle les conditions à gauche puis clique sur **Prédire**.")

with tab_perf:
    st.subheader("Comparaison des modèles (R² sur test)")
    comp = METRICS if METRICS else {}
    if comp:
        rows = [{"Modèle": n.replace("_", " ").title(),
                 "R² test": round(m["test_metrics"]["r2"], 3),
                 "RMSE": round(m["test_metrics"]["rmse"]),
                 "MAE": round(m["test_metrics"]["mae"])}
                for n, m in sorted(comp.items(), key=lambda kv: -kv[1]["test_metrics"]["r2"])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "🔑 La consommation étant fortement autocorrélée (0,90 avec la veille), "
        "l'ajout de variables autorégressives (lags) fait passer le R² test de "
        "~0,53 (météo seule) à **0,86**."
    )

    col1, col2 = st.columns(2)
    for col, img, cap in [(col1, "model_comparison.png", "Comparaison des modèles"),
                          (col2, "feature_importance.png", "Importance des variables")]:
        p = os.path.join(RESULTS_DIR, img)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)
    pva = os.path.join(RESULTS_DIR, "predictions_vs_actual.png")
    if os.path.exists(pva):
        st.image(pva, caption="Consommation réelle vs prédite (test)", use_container_width=True)

st.divider()
st.caption("Alexis Clerc · [GitHub](https://github.com/2Alexis) · [Portfolio](https://alexis-clerc.fr)")
