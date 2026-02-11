import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def create_output_directory(base_dir='../visualizations'):
    """
    Crée un répertoire pour les graphiques
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'{base_dir}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Fonctions de visualisation des résultats du modèle

def load_comparison_data():
    """
    Charge le fichier de comparaison le plus récent
    """
    comparison_files = glob.glob('../evaluations/comparison_*.csv')
    if not comparison_files:
        print("Erreur: Aucun fichier de comparaison trouvé.")
        return None
    
    # Trouver le fichier de comparaison le plus récent
    latest_comparison = max(comparison_files, key=os.path.getctime)
    print(f"Chargement des données de comparaison: {latest_comparison}")
    
    return pd.read_csv(latest_comparison)

def plot_actual_vs_predicted(data, output_dir):
    """
    Crée un graphique des valeurs réelles vs prédites
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data['valeur_reelle'], data['prediction'], alpha=0.5)
    
    # Ajouter une ligne de référence y=x
    min_val = min(data['valeur_reelle'].min(), data['prediction'].min())
    max_val = max(data['valeur_reelle'].max(), data['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Valeurs réelles (Wh)')
    plt.ylabel('Prédictions (MWh)')
    plt.title('Valeurs réelles vs Prédictions')
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé: {output_dir}/actual_vs_predicted.png")

def plot_error_distribution(data, output_dir):
    """
    Crée un histogramme de la distribution des erreurs
    """
    plt.figure(figsize=(10, 8))
    
    # Histogramme des erreurs
    sns.histplot(data['erreur'], kde=True)
    
    plt.xlabel('Erreur (MWh)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs')
    plt.grid(True)
    
    # Ajouter une ligne verticale à zéro
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé: {output_dir}/error_distribution.png")

def plot_error_percentage_distribution(data, output_dir):
    """
    Crée un histogramme de la distribution des erreurs en pourcentage
    """
    plt.figure(figsize=(10, 8))
    
    # Filtrer les valeurs aberrantes
    filtered_data = data[(data['erreur_pourcentage'] > -50) & (data['erreur_pourcentage'] < 50)]
    
    # Histogramme des erreurs en pourcentage
    sns.histplot(filtered_data['erreur_pourcentage'], kde=True)
    
    plt.xlabel('Erreur (%)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs en pourcentage')
    plt.grid(True)
    
    # Ajouter une ligne verticale à zéro
    plt.axvline(x=0, color='r', linestyle='--')
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_percentage_distribution.png', dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé: {output_dir}/error_percentage_distribution.png")

def plot_residuals_vs_predicted(data, output_dir):
    """
    Crée un graphique des résidus vs valeurs prédites
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data['prediction'], data['erreur'], alpha=0.5)
    
    # Ajouter une ligne horizontale à y=0
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Valeurs prédites (MWh)')
    plt.ylabel('Résidus (MWh)')
    plt.title('Résidus vs Valeurs prédites')
    plt.grid(True)
    
    # Sauvegarder le graphique
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals_vs_predicted.png', dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé: {output_dir}/residuals_vs_predicted.png")

def visualize_results():
    """
    Visualise les résultats du modèle
    """
    print("Démarrage de la visualisation des résultats...")
    
    # Charger les données de comparaison
    data = load_comparison_data()
    if data is None:
        return
    
    # Créer le répertoire de sortie
    output_dir = create_output_directory()
    
    # Créer les graphiques
    plot_actual_vs_predicted(data, output_dir)
    plot_error_distribution(data, output_dir)
    plot_error_percentage_distribution(data, output_dir)
    plot_residuals_vs_predicted(data, output_dir)
    
    print("Visualisation terminée!")

# Fonctions de visualisation générales pour l'analyse exploratoire

def plot_energy_consumption_over_time(df, date_column='date', consumption_column='consumption', output_dir='../results'):
    """
    Trace la consommation d'énergie au fil du temps
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df[date_column], df[consumption_column])
    plt.title('Consommation d\'énergie au fil du temps')
    plt.xlabel('Date')
    plt.ylabel('Consommation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/consumption_over_time.png')
    plt.close()

def plot_correlation_matrix(df, output_dir='../results'):
    """
    Trace une matrice de corrélation des features numériques
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation des features')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, output_dir='../results'):
    """
    Trace l'importance des features pour les modèles qui le supportent
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Importance des features')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()

def plot_predictions_vs_actual(y_true, y_pred, output_dir='../results', title='Prédictions vs Valeurs réelles'):
    """
    Trace les prédictions vs les valeurs réelles
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictions_vs_actual.png')
    plt.close()

def plot_residuals(y_true, y_pred, output_dir='../results'):
    """
    Trace l'analyse des résidus
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 4))
    
    # Distribution des résidus
    plt.subplot(121)
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des résidus')
    plt.xlabel('Résidus')
    
    # Résidus vs Prédictions
    plt.subplot(122)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Résidus vs Prédictions')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residuals_analysis.png')
    plt.close()

# Fonctions pour créer des graphiques de performance d'entraînement

def create_training_performance_plots(model_name, train_metrics, test_metrics=None, output_dir=None):
    """
    Crée des graphiques de performance pour un modèle
    
    Args:
        model_name (str): Nom du modèle
        train_metrics (dict): Métriques sur l'ensemble d'entraînement
        test_metrics (dict, optional): Métriques sur l'ensemble de test
        output_dir (str, optional): Répertoire de sortie
    """
    if output_dir is None:
        output_dir = create_output_directory('../performance_plots')
    
    # Créer le graphique de comparaison des métriques
    plt.figure(figsize=(12, 6))
    
    metrics = ['rmse', 'mae', 'r2']
    x = np.arange(len(metrics))
    width = 0.35
    
    train_values = [train_metrics[m] for m in metrics]
    
    rects1 = plt.bar(x - width/2, train_values, width, label='Entraînement')
    
    if test_metrics:
        test_values = [test_metrics[m] for m in metrics]
        rects2 = plt.bar(x + width/2, test_values, width, label='Test')
    
    plt.xlabel('Métriques')
    plt.ylabel('Valeurs')
    plt.title(f'Performance du modèle {model_name}')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_metrics_comparison.png')
    plt.close()
    
    print(f"Graphique de performance sauvegardé: {output_dir}/{model_name}_metrics_comparison.png")

if __name__ == "__main__":
    # Menu simple pour choisir l'action à effectuer
    print("Module de visualisation")
    print("1. Visualiser les résultats du modèle")
    
    choice = input("Choisissez une option (1): ")
    
    if choice == "1":
        visualize_results()
    else:
        print("Option invalide") 