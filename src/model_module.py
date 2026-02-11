import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import glob
from datetime import datetime

# Fonctions pour l'entraînement des modèles

def load_data():
    """
    Charge les données d'entraînement et de test
    """
    train_data = pd.read_csv('../data/processed/train_data_final.csv')
    test_data = pd.read_csv('../data/processed/test_data_final.csv')
    
    # Séparer les features et la cible
    features = [col for col in train_data.columns if col != 'consommation']
    X_train = train_data[features]
    y_train = train_data['consommation']
    X_test = test_data[features]
    
    # Vérifier si la colonne consommation existe dans les données de test
    if 'consommation' in test_data.columns:
        y_test = test_data['consommation']
    else:
        print("ATTENTION: La colonne 'consommation' n'existe pas dans les données de test.")
        print("Les métriques sur l'ensemble de test ne seront pas calculées.")
        y_test = None
    
    # Gérer les valeurs manquantes
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Utiliser la moyenne de l'ensemble d'entraînement
    y_train = y_train.fillna(y_train.mean())
    
    if y_test is not None:
        y_test = y_test.fillna(y_train.mean())  # Utiliser la moyenne de l'ensemble d'entraînement
    
    return X_train, X_test, y_train, y_test, features

def train_models(X_train, y_train, X_test, y_test):
    """
    Entraîne plusieurs modèles et évalue leurs performances
    """
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEntraînement du modèle {name}...")
        model.fit(X_train, y_train)
        
        # Prédictions sur l'ensemble d'entraînement
        y_pred_train = model.predict(X_train)
        
        # Métriques d'évaluation sur l'ensemble d'entraînement
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        results[name] = {
            'train': train_metrics,
            'model': model
        }
        
        print(f"Performances sur l'ensemble d'entraînement:")
        print(f"RMSE: {train_metrics['rmse']:.2f}")
        print(f"MAE: {train_metrics['mae']:.2f}")
        print(f"R²: {train_metrics['r2']:.4f}")
        
        # Prédictions et métriques sur l'ensemble de test si disponible
        if y_test is not None:
            y_pred_test = model.predict(X_test)
            test_metrics = {
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
            results[name]['test'] = test_metrics
            
            print(f"Performances sur l'ensemble de test:")
            print(f"RMSE: {test_metrics['rmse']:.2f}")
            print(f"MAE: {test_metrics['mae']:.2f}")
            print(f"R²: {test_metrics['r2']:.4f}")
    
    return results

def analyze_feature_importance(model, features):
    """
    Analyse l'importance des features pour le modèle
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
    return None

def save_results(results, features):
    """
    Sauvegarde les modèles et leurs résultats
    """
    # Créer le dossier models s'il n'existe pas
    os.makedirs('../models', exist_ok=True)
    
    # Trouver le meilleur modèle basé sur le R² sur l'ensemble d'entraînement
    # (si les données de test ne sont pas disponibles)
    if 'test' in next(iter(results.values())):
        best_model_name = max(results.keys(), key=lambda k: results[k]['test']['r2'])
        metric_source = 'test'
    else:
        best_model_name = max(results.keys(), key=lambda k: results[k]['train']['r2'])
        metric_source = 'train'
    
    best_model = results[best_model_name]['model']
    
    # Sauvegarder le meilleur modèle
    model_path = f'../models/best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    joblib.dump(best_model, model_path)
    
    # Sauvegarder les métriques
    metrics = {}
    for name in results.keys():
        metrics[name] = {'train_metrics': results[name]['train']}
        if 'test' in results[name]:
            metrics[name]['test_metrics'] = results[name]['test']
    
    metrics_path = f'../models/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Analyser l'importance des features pour le meilleur modèle
    importance = analyze_feature_importance(best_model, features)
    if importance is not None:
        importance_path = f'../models/feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        importance.to_csv(importance_path, index=False)
    
    print(f"\nMeilleur modèle ({best_model_name}) sauvegardé: {model_path}")
    print(f"Sélectionné sur la base des métriques de {metric_source}")
    print(f"Métriques sauvegardées: {metrics_path}")
    if importance is not None:
        print(f"Importance des features sauvegardée: {importance_path}")

def train_model():
    """
    Fonction principale pour l'entraînement et l'évaluation des modèles
    """
    print("Chargement des données...")
    X_train, X_test, y_train, y_test, features = load_data()
    
    # Si y_test est None, essayer de charger les valeurs réelles depuis test_data_raw.csv
    if y_test is None:
        try:
            test_raw_path = '../data/processed/test_data_raw.csv'
            if os.path.exists(test_raw_path):
                test_raw = pd.read_csv(test_raw_path)
                if 'consommation' in test_raw.columns:
                    print("Utilisation des valeurs de consommation depuis test_data_raw.csv pour l'évaluation")
                    y_test = test_raw['consommation']
        except Exception as e:
            print(f"Erreur lors du chargement des données de test brutes: {e}")
    
    print("\nEntraînement des modèles...")
    results = train_models(X_train, y_train, X_test, y_test)
    
    print("\nSauvegarde des résultats...")
    save_results(results, features)
    
    print("\nEntraînement terminé!")

# Fonctions pour l'évaluation des modèles

def evaluate_model():
    """
    Évalue les performances du modèle sur les données de test
    """
    # Charger les prédictions
    predictions = load_predictions()
    if predictions is None:
        return
    
    # Charger les valeurs réelles
    actual_values = load_actual_values()
    if actual_values is None:
        return
    
    # Fusionner les données
    comparison_data = pd.merge(
        predictions, 
        actual_values, 
        left_index=True, 
        right_index=True,
        suffixes=('_pred', '_actual')
    )
    
    # Renommer les colonnes
    comparison_data = comparison_data.rename(columns={
        'consommation_pred': 'prediction',
        'consommation_actual': 'valeur_reelle'
    })
    
    # Calculer les erreurs
    comparison_data['erreur'] = comparison_data['valeur_reelle'] - comparison_data['prediction']
    comparison_data['erreur_pourcentage'] = (comparison_data['erreur'] / comparison_data['valeur_reelle']) * 100
    
    # Calculer les métriques
    mse = mean_squared_error(comparison_data['valeur_reelle'], comparison_data['prediction'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(comparison_data['valeur_reelle'], comparison_data['prediction'])
    r2 = r2_score(comparison_data['valeur_reelle'], comparison_data['prediction'])
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs('../evaluations', exist_ok=True)
    
    # Enregistrer les métriques
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f'../evaluations/metrics_{timestamp}.txt'
    
    with open(metrics_path, 'w') as f:
        f.write("Évaluation du modèle sur les données de test\n")
        f.write("==========================================\n\n")
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R²: {r2:.4f}\n")
    
    # Enregistrer les données de comparaison
    comparison_path = f'../evaluations/comparison_{timestamp}.csv'
    comparison_data.to_csv(comparison_path, index=False)
    
    print(f"Métriques d'évaluation enregistrées: {metrics_path}")
    print(f"Données de comparaison enregistrées: {comparison_path}")
    print("\nRésultats de l'évaluation:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

# Fonctions pour faire des prédictions

def load_latest_model():
    """
    Charge le modèle le plus récent
    """
    model_files = glob.glob('../models/best_model_*.joblib')
    if not model_files:
        print("Erreur: Aucun modèle trouvé.")
        return None
    
    # Trouver le modèle le plus récent
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Chargement du modèle: {latest_model}")
    
    return joblib.load(latest_model)

def load_test_data():
    """
    Charge les données de test
    """
    test_path = '../data/processed/test_data_final.csv'
    if not os.path.exists(test_path):
        print(f"Erreur: Le fichier {test_path} n'existe pas.")
        return None
    
    # Charger les données
    test_data = pd.read_csv(test_path)
    
    return test_data

def load_predictions():
    """
    Charge le fichier de prédictions le plus récent
    """
    prediction_files = glob.glob('../predictions/predictions_*.csv')
    if not prediction_files:
        print("Erreur: Aucun fichier de prédictions trouvé.")
        return None
    
    # Trouver le fichier de prédictions le plus récent
    latest_prediction = max(prediction_files, key=os.path.getctime)
    print(f"Chargement des prédictions: {latest_prediction}")
    
    return pd.read_csv(latest_prediction)

def load_actual_values():
    """
    Charge les valeurs réelles des données de test
    """
    # Charger le fichier de test avec les valeurs réelles
    test_path = '../data/processed/test_data_raw.csv'
    if not os.path.exists(test_path):
        print(f"Erreur: Le fichier {test_path} n'existe pas.")
        return None
    
    # Charger les données
    test_data = pd.read_csv(test_path)
    
    # Vérifier si la colonne consommation existe
    if 'consommation' not in test_data.columns:
        print("Erreur: La colonne 'consommation' n'existe pas dans les données de test.")
        return None
    
    # Extraire les valeurs réelles
    actual_values = test_data[['consommation']]
    
    return actual_values

def predict():
    """
    Fait des prédictions avec le modèle entraîné
    """
    # Charger le modèle
    model = load_latest_model()
    if model is None:
        return
    
    # Charger les données de test
    test_data = load_test_data()
    if test_data is None:
        return
    
    # Faire des prédictions
    print("Prédiction en cours...")
    predictions = model.predict(test_data)
    
    # Créer un DataFrame avec les prédictions
    predictions_df = pd.DataFrame({
        'consommation': predictions
    })
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs('../predictions', exist_ok=True)
    
    # Enregistrer les prédictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = f'../predictions/predictions_{timestamp}.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"Prédictions enregistrées: {predictions_path}")

if __name__ == "__main__":
    # Menu simple pour choisir l'action à effectuer
    print("Module de modélisation")
    print("1. Entraîner un modèle")
    print("2. Faire des prédictions")
    print("3. Évaluer le modèle")
    
    choice = input("Choisissez une option (1-3): ")
    
    if choice == "1":
        train_model()
    elif choice == "2":
        predict()
    elif choice == "3":
        evaluate_model()
    else:
        print("Option invalide") 