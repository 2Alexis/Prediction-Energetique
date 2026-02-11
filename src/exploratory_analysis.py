import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
plt.style.use('default')

def load_and_prepare_data():
    """
    Charge et prépare les données avec des features additionnelles
    """
    df = pd.read_csv('data/raw/cleaned_file.csv')
    
    # Renommer les colonnes pour plus de clarté
    df.columns = ['date_debut', 'date_fin', 'consommation', 'temperature', 'humidite', 
                 'vitesse_vent', 'precipitation', 'evenement']
    
    # Convertir les dates en format datetime
    df['date_debut'] = pd.to_datetime(df['date_debut'])
    df['date_fin'] = pd.to_datetime(df['date_fin'])
    
    # Ajouter des features temporelles
    df['annee'] = df['date_debut'].dt.year
    df['mois'] = df['date_debut'].dt.month
    df['jour'] = df['date_debut'].dt.day
    df['jour_semaine'] = df['date_debut'].dt.dayofweek
    df['weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
    df['saison'] = df['date_debut'].dt.month.map({12:1, 1:1, 2:1,  # Hiver
                                                 3:2, 4:2, 5:2,    # Printemps
                                                 6:3, 7:3, 8:3,    # Été
                                                 9:4, 10:4, 11:4}) # Automne
    
    return df

def analyze_seasonality(df):
    """
    Analyse la saisonnalité de la consommation
    """
    # Consommation moyenne par saison
    seasonal_consumption = df.groupby('saison')['consommation'].agg(['mean', 'std']).round(2)
    seasonal_consumption.index = ['Hiver', 'Printemps', 'Été', 'Automne']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    seasonal_consumption['mean'].plot(kind='bar')
    plt.title('Consommation moyenne par saison')
    plt.xlabel('Saison')
    plt.ylabel('Consommation moyenne (Wh)')
    
    # Consommation moyenne par mois
    monthly_consumption = df.groupby('mois')['consommation'].mean()
    plt.subplot(1, 2, 2)
    monthly_consumption.plot(kind='line', marker='o')
    plt.title('Consommation moyenne par mois')
    plt.xlabel('Mois')
    plt.ylabel('Consommation moyenne (Wh)')
    plt.tight_layout()
    plt.savefig('results/seasonality_analysis.png')
    plt.close()
    
    print("\nConsommation moyenne par saison:")
    print(seasonal_consumption)

def analyze_temperature_impact(df):
    """
    Analyse l'impact de la température sur la consommation
    """
    plt.figure(figsize=(15, 5))
    
    # Relation température-consommation
    plt.subplot(1, 2, 1)
    plt.scatter(df['temperature'], df['consommation'], alpha=0.5)
    plt.title('Relation Température-Consommation')
    plt.xlabel('Température (°C)')
    plt.ylabel('Consommation (Wh)')
    
    # Consommation moyenne par plage de température
    df['temp_range'] = pd.cut(df['temperature'], bins=10)
    temp_consumption = df.groupby('temp_range')['consommation'].mean()
    
    plt.subplot(1, 2, 2)
    temp_consumption.plot(kind='bar')
    plt.title('Consommation moyenne par plage de température')
    plt.xlabel('Plage de température')
    plt.ylabel('Consommation moyenne (Wh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/temperature_impact.png')
    plt.close()

def analyze_peak_consumption(df):
    """
    Analyse les périodes de pic de consommation
    """
    # Identifier les pics de consommation (au-dessus du 90e percentile)
    threshold = df['consommation'].quantile(0.9)
    peak_days = df[df['consommation'] > threshold]
    
    # Distribution des pics par jour de la semaine
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    peak_days['jour_semaine'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution des pics par jour de la semaine')
    plt.xlabel('Jour de la semaine (0=Lundi)')
    plt.ylabel('Nombre de pics')
    
    # Distribution des pics par mois
    plt.subplot(1, 2, 2)
    peak_days['mois'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution des pics par mois')
    plt.xlabel('Mois')
    plt.ylabel('Nombre de pics')
    plt.tight_layout()
    plt.savefig('results/peak_consumption.png')
    plt.close()
    
    print("\nAnalyse des pics de consommation:")
    print(f"Seuil de pic: {threshold:.2f} Wh")
    print(f"Nombre de jours de pic: {len(peak_days)}")
    print("\nCaractéristiques moyennes des jours de pic:")
    print(peak_days[['temperature', 'humidite', 'vitesse_vent', 'precipitation']].mean())

def analyze_weather_patterns(df):
    """
    Analyse approfondie des patterns météorologiques
    """
    weather_cols = ['temperature', 'humidite', 'vitesse_vent', 'precipitation']
    
    # Créer une matrice de scatter plots
    plt.figure(figsize=(12, 12))
    sns.pairplot(df[weather_cols + ['consommation']], diag_kind='kde')
    plt.tight_layout()
    plt.savefig('results/weather_patterns.png')
    plt.close()
    
    # Calculer les corrélations conditionnelles
    high_temp = df[df['temperature'] > df['temperature'].median()]
    low_temp = df[df['temperature'] <= df['temperature'].median()]
    
    print("\nCorrélations avec la consommation:")
    print("\nJours chauds (température > médiane):")
    print(high_temp[weather_cols].corrwith(high_temp['consommation']).round(3))
    print("\nJours froids (température <= médiane):")
    print(low_temp[weather_cols].corrwith(low_temp['consommation']).round(3))

def prepare_model_data(df):
    """
    Prépare les données pour la modélisation
    """
    # Sélectionner les features pertinentes
    features = ['temperature', 'humidite', 'vitesse_vent', 'precipitation', 
                'mois', 'jour_semaine', 'weekend', 'evenement']
    X = df[features]
    y = df['consommation']
    
    # Gérer les valeurs manquantes
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser uniquement les features numériques continues
    scaler = StandardScaler()
    numerical_features = ['temperature', 'humidite', 'vitesse_vent', 'precipitation']
    
    # Créer des copies pour éviter de modifier les originaux
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Normaliser uniquement les features numériques
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Sauvegarder les données préparées
    train_data = X_train_scaled.copy()
    train_data['consommation'] = y_train
    test_data = X_test_scaled.copy()
    test_data['consommation'] = y_test
    
    # Créer le dossier data/processed s'il n'existe pas
    os.makedirs('data/processed', exist_ok=True)
    
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    print("\nDonnées préparées pour la modélisation:")
    print(f"Ensemble d'entraînement: {X_train.shape}")
    print(f"Ensemble de test: {X_test.shape}")

def main():
    """
    Fonction principale pour l'analyse exploratoire approfondie
    """
    # Créer les dossiers nécessaires
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Charger et préparer les données
    print("Chargement et préparation des données...")
    df = load_and_prepare_data()
    
    # Analyses approfondies
    print("\nAnalyse de la saisonnalité...")
    analyze_seasonality(df)
    
    print("\nAnalyse de l'impact de la température...")
    analyze_temperature_impact(df)
    
    print("\nAnalyse des pics de consommation...")
    analyze_peak_consumption(df)
    
    print("\nAnalyse des patterns météorologiques...")
    analyze_weather_patterns(df)
    
    print("\nPréparation des données pour la modélisation...")
    prepare_model_data(df)
    
    print("\nAnalyse exploratoire approfondie terminée!")
    print("Tous les résultats ont été sauvegardés dans le dossier 'results'")
    print("Les données préparées pour la modélisation sont dans 'data/processed'")

if __name__ == "__main__":
    main() 