# Prédiction de la Demande Énergétique Quotidienne

Ce projet vise à développer un modèle de machine learning capable de prédire la demande énergétique quotidienne d'une ville en fonction de divers paramètres tels que la météo, la consommation passée et les événements spéciaux.

## Structure du Projet

```
project_ia/
│
├── data/               # Dossier contenant les données
│   ├── raw/           # Données brutes
│   └── processed/     # Données traitées
│
├── models/            # Modèles entraînés et métriques
│
├── notebooks/         # Notebooks Jupyter pour l'analyse exploratoire
│
├── results/           # Visualisations et résultats
│
└── src/              # Code source
    ├── data_preparation.py    # Préparation des données
    ├── model_training.py      # Entraînement des modèles
    └── visualization.py       # Visualisation des données et résultats
```

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd project_ia
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Préparation des données :
```bash
python src/data_preprocessing.py
```

2. Entraînement du modèle :
```bash
python src/model_training.py
```

3. Visualisation des résultats :
```bash
python src/visualization.py
```

## Fonctionnalités

- Nettoyage et préparation des données
- Feature engineering pour les variables temporelles
- Entraînement de différents modèles (Random Forest, Régression Linéaire)
- Évaluation des modèles avec diverses métriques
- Visualisations des résultats et analyses

## Métriques d'Évaluation

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Prochaines Étapes

- [ ] Intégration de données météorologiques
- [ ] Ajout de nouveaux modèles (XGBoost, LightGBM)
- [ ] Optimisation des hyperparamètres
- [ ] Déploiement du modèle via une API Flask
- [ ] Interface utilisateur pour les prédictions

## Auteur

[Votre Nom]

## Licence

Ce projet est sous licence MIT. 