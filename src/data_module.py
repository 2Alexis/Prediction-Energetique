import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import glob
from datetime import datetime

def split_data(normalize=True):
    """
    Divise les données prétraitées en ensembles d'entraînement (80%) et de test (20%)
    avec option de normalisation
    
    Args:
        normalize (bool): Si True, normalise les données numériques
    """
    # Vérifier si le fichier existe
    data_path = '../data/processed/energy_data_processed.csv'
    if not os.path.exists(data_path):
        print(f"Erreur: Le fichier {data_path} n'existe pas.")
        return
    
    # Charger les données prétraitées
    print("Chargement des données...")
    df = pd.read_csv(data_path)
    
    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement (80%) et de test (20%)...")
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Définir les chemins de sortie
    if normalize:
        train_path = '../data/processed/train_data.csv'
        test_path = '../data/processed/test_data.csv'
    else:
        train_path = '../data/processed/train_data_raw.csv'
        test_path = '../data/processed/test_data_raw.csv'
    
    # Normaliser les données si demandé
    if normalize:
        print("Normalisation des données...")
        # Séparer les features numériques à normaliser
        numerical_features = ['temperature', 'humidite', 'vitesse_vent', 'precipitation']
        
        # Créer un scaler et l'appliquer uniquement aux features numériques
        scaler = StandardScaler()
        
        # Normaliser les features numériques dans l'ensemble d'entraînement
        train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
        
        # Normaliser les features numériques dans l'ensemble de test en utilisant le scaler entraîné
        test_data[numerical_features] = scaler.transform(test_data[numerical_features])
    
    # Enregistrer les ensembles d'entraînement et de test
    print(f"Enregistrement des données d'entraînement: {train_path}")
    train_data.to_csv(train_path, index=False)
    
    print(f"Enregistrement des données de test: {test_path}")
    test_data.to_csv(test_path, index=False)
    
    print("Division des données terminée!")

def prepare_data_for_model(remove_target_from_test=False):
    """
    Prépare les données pour l'entraînement du modèle
    
    Args:
        remove_target_from_test (bool): Si True, supprime la colonne cible des données de test
    """
    # Vérifier si les fichiers existent
    train_path = '../data/processed/train_data_raw.csv'
    test_path = '../data/processed/test_data_raw.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Erreur: Les fichiers d'entraînement ou de test n'existent pas.")
        return
    
    # Charger les données
    print("Chargement des données brutes...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Renommer les fichiers finaux pour l'entraînement du modèle
    print("Préparation des données pour le modèle...")
    
    # Enregistrer les données
    train_output = '../data/processed/train_data_final.csv'
    test_output = '../data/processed/test_data_final.csv'
    
    # Supprimer la colonne cible des données de test si demandé
    if remove_target_from_test and 'consommation' in test_data.columns:
        print("Suppression de la colonne 'consommation' des données de test...")
        test_data = test_data.drop('consommation', axis=1)
    
    # Enregistrer les données
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)
    
    print(f"Données d'entraînement enregistrées: {train_output}")
    print(f"Données de test enregistrées: {test_output}")
    print("Préparation terminée!")

def preprocess_data():
    """
    Prétraite les données brutes
    """
    # Fonctions de prétraitement à implémenter selon les besoins
    print("Prétraitement des données non implémenté. Utilisez data_preprocessing.py.")

if __name__ == "__main__":
    # Menu simple pour choisir l'action à effectuer
    print("Module de gestion des données")
    print("1. Diviser les données (avec normalisation)")
    print("2. Diviser les données (sans normalisation)")
    print("3. Préparer les données pour le modèle")
    print("4. Préparer les données et supprimer la cible des données de test")
    
    choice = input("Choisissez une option (1-4): ")
    
    if choice == "1":
        split_data(normalize=True)
    elif choice == "2":
        split_data(normalize=False)
    elif choice == "3":
        prepare_data_for_model(remove_target_from_test=False)
    elif choice == "4":
        prepare_data_for_model(remove_target_from_test=True)
    else:
        print("Option invalide") 