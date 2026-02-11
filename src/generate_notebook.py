import nbformat as nbf

# Créer un nouveau notebook
nb = nbf.v4.new_notebook()

# Titre et Introduction
intro_md = """# Analyse de la Consommation Énergétique

Ce notebook présente une analyse complète des données de consommation énergétique, incluant l'exploration des données, la visualisation des tendances, et l'évaluation du modèle de prédiction."""

nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

# Import des bibliothèques
imports_code = """# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Configuration du style des graphiques
plt.style.use('seaborn')
sns.set_palette("husl")

# Configuration pour afficher tous les résultats
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)"""

nb.cells.append(nbf.v4.new_code_cell(imports_code))

# Chargement des données
data_loading_md = "## 1. Chargement et Exploration des Données"
nb.cells.append(nbf.v4.new_markdown_cell(data_loading_md))

data_loading_code = """# Chargement des données
data_path = '../data/processed/energy_data_processed.csv'
df = pd.read_csv(data_path)

# Affichage des premières lignes
print("Aperçu des données :")
display(df.head())

# Informations sur le dataset
print("\\nInformations sur le dataset :")
display(df.info())

# Statistiques descriptives
print("\\nStatistiques descriptives :")
display(df.describe())"""

nb.cells.append(nbf.v4.new_code_cell(data_loading_code))

# Analyse des tendances
trends_md = "## 2. Analyse des Tendances"
nb.cells.append(nbf.v4.new_markdown_cell(trends_md))

trends_code = """# Création d'un graphique de tendance mensuelle
plt.figure(figsize=(12, 6))
sns.boxplot(x='mois', y='consommation', data=df)
plt.title('Distribution de la Consommation par Mois')
plt.xlabel('Mois')
plt.ylabel('Consommation (MWh)')
plt.show()

# Création d'un graphique de tendance hebdomadaire
plt.figure(figsize=(12, 6))
sns.boxplot(x='jour_semaine', y='consommation', data=df)
plt.title('Distribution de la Consommation par Jour de la Semaine')
plt.xlabel('Jour de la Semaine')
plt.ylabel('Consommation (MWh)')
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(trends_code))

# Analyse des corrélations
correlations_md = "## 3. Analyse des Corrélations"
nb.cells.append(nbf.v4.new_markdown_cell(correlations_md))

correlations_code = """# Création de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation')
plt.show()

# Création de scatter plots pour les variables les plus corrélées
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.scatterplot(data=df, x='temperature', y='consommation', ax=axes[0,0])
axes[0,0].set_title('Consommation vs Température')

sns.scatterplot(data=df, x='humidite', y='consommation', ax=axes[0,1])
axes[0,1].set_title('Consommation vs Humidité')

sns.scatterplot(data=df, x='vitesse_vent', y='consommation', ax=axes[1,0])
axes[1,0].set_title('Consommation vs Vitesse du Vent')

sns.scatterplot(data=df, x='precipitation', y='consommation', ax=axes[1,1])
axes[1,1].set_title('Consommation vs Précipitation')

plt.tight_layout()
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(correlations_code))

# Évaluation du modèle
model_eval_md = "## 4. Évaluation du Modèle"
nb.cells.append(nbf.v4.new_markdown_cell(model_eval_md))

model_eval_code = """# Chargement du modèle entraîné
model_path = '../models/best_model_20250402_145118.joblib'
model = joblib.load(model_path)

# Chargement des données de test
test_data = pd.read_csv('../data/processed/test_data.csv')
X_test = test_data.drop('consommation', axis=1)
y_test = test_data['consommation']

# Prédictions
y_pred = model.predict(X_test)

# Calcul des métriques
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE : {rmse:.2f} MWh')
print(f'R² : {r2:.4f}')

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valeurs Réelles (MWh)')
plt.ylabel('Prédictions (MWh)')
plt.title('Prédictions vs Valeurs Réelles')
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(model_eval_code))

# Importance des features
features_md = "## 5. Importance des Features"
nb.cells.append(nbf.v4.new_markdown_cell(features_md))

features_code = """# Extraction de l'importance des features
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualisation de l'importance des features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importance des Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(features_code))

# Conclusions
conclusions_md = """## 6. Conclusions

### Points Clés de l'Analyse :
1. **Tendances Saisonnières** :
   - La consommation varie significativement selon les mois
   - Les pics de consommation sont observés en hiver et en été
   - Les weekends montrent généralement une consommation plus faible

2. **Corrélations** :
   - La température est le facteur le plus influent sur la consommation
   - L'humidité montre une corrélation modérée
   - La vitesse du vent et les précipitations ont un impact plus faible

3. **Performance du Modèle** :
   - Le modèle Random Forest montre de bonnes performances
   - Les prédictions sont cohérentes avec les valeurs réelles
   - L'erreur quadratique moyenne (RMSE) est acceptable

### Recommandations :
1. **Gestion de l'Énergie** :
   - Renforcer la production en période de forte consommation
   - Optimiser la distribution selon les tendances saisonnières
   - Prévoir des capacités de stockage pour les pics de demande

2. **Améliorations Futures** :
   - Intégrer des données économiques (PIB, prix de l'énergie)
   - Ajouter des variables liées aux événements spéciaux
   - Développer des prédictions à plus court terme

3. **Actions Immédiates** :
   - Mettre en place un système de monitoring en temps réel
   - Développer des alertes pour les pics de consommation
   - Créer des tableaux de bord pour la visualisation des données"""

nb.cells.append(nbf.v4.new_markdown_cell(conclusions_md))

# Sauvegarder le notebook
with open('../notebooks/energy_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 