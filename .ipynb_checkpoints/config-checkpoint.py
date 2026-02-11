# Configuration du projet

# Chemins des fichiers
DATA_RAW_PATH = 'data/raw/cleaned_file.csv'
DATA_PROCESSED_PATH = 'data/processed/energy_data.csv'
MODEL_PATH = 'models/rf_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
METRICS_PATH = 'models/rf_metrics.json'

# Paramètres du modèle
MODEL_PARAMS = {
    'rf': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
}

# Paramètres de préparation des données
DATA_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'outlier_std': 3
}

# Configuration de l'API
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# Features à utiliser pour la prédiction
FEATURES = [
    'year',
    'month',
    'day',
    'day_of_week',
    'is_weekend'
]

# Colonne cible
TARGET_COLUMN = 'Consumption' 