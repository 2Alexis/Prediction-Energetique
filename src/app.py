from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import json
import glob
from datetime import datetime

app = Flask(__name__)

# Fonction pour charger le modèle le plus récent
def load_latest_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_files = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier models/")
    
    latest_model = max(model_files)
    model_path = os.path.join(models_dir, latest_model)
    return joblib.load(model_path), latest_model.replace('best_model_', '').replace('.joblib', '')

# Fonction pour charger les métriques les plus récentes
def load_latest_metrics():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    metrics_files = [f for f in os.listdir(models_dir) if f.startswith('metrics_') and f.endswith('.json')]
    if not metrics_files:
        return None
    
    latest_metrics_file = max(metrics_files)
    metrics_path = os.path.join(models_dir, latest_metrics_file)
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

# Fonction pour charger l'importance des features
def load_feature_importance():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    importance_files = glob.glob(os.path.join(models_dir, 'feature_importance_*.csv'))
    if not importance_files:
        return None
    
    latest_importance = max(importance_files, key=os.path.getctime)
    
    import pandas as pd
    importance_df = pd.read_csv(latest_importance)
    return importance_df.to_dict('records')

# Charger le modèle
try:
    model, model_timestamp = load_latest_model()
    print(f"Modèle chargé avec succès : {model}")
    
    # Charger les métriques
    metrics = load_latest_metrics()
    if metrics:
        print("Métriques chargées avec succès")
    else:
        print("Aucune métrique trouvée")
    
    # Charger l'importance des features
    feature_importance = load_feature_importance()
    if feature_importance:
        print("Importance des features chargée avec succès")
    else:
        print("Aucune importance des features trouvée")
        
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    model = None
    metrics = None
    feature_importance = None
    model_timestamp = None

@app.route('/')
def home():
    return render_template('index.html', model_timestamp=model_timestamp)

@app.route('/comparison')
def comparison():
    model_data = {}
    
    if metrics:
        for model_name, model_metrics in metrics.items():
            model_data[model_name] = {
                'train': {
                    'rmse': model_metrics['train_metrics']['rmse'],
                    'mae': model_metrics['train_metrics']['mae'],
                    'r2': model_metrics['train_metrics']['r2']
                }
            }
            
            if 'test_metrics' in model_metrics:
                model_data[model_name]['test'] = {
                    'rmse': model_metrics['test_metrics']['rmse'],
                    'mae': model_metrics['test_metrics']['mae'],
                    'r2': model_metrics['test_metrics']['r2']
                }
    
    return render_template('comparison.html', 
                          model_data=model_data, 
                          feature_importance=feature_importance,
                          model_timestamp=model_timestamp)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Le modèle n\'est pas disponible'}), 500
    
    try:
        # Récupérer les données du formulaire
        data = request.get_json()
        
        # Créer le vecteur de features dans le bon ordre
        features = np.array([[
            data['temperature'],
            data['humidite'],
            data['vitesse_vent'],
            data['precipitation'],
            data['mois'],
            data['jour_semaine'],
            data['weekend'],
            data['evenement']
        ]])
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': float(prediction)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 