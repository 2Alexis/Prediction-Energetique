import pandas as pd
import numpy as np
import os
from datetime import datetime

def preprocess_data():
    """Load and preprocess the raw data"""
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load the raw data
    df = pd.read_csv('data/raw/cleaned_file.csv')
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'Temperature': 'temperature',
        'Humidity': 'humidite',
        'Wind Speed': 'vitesse_vent',
        'Precipitation': 'precipitation',
        'Consommation(W)': 'consommation'
    })
    
    # Create month and day of week from date
    df['mois'] = pd.to_datetime(df['Date_Debut_x']).dt.month
    df['jour_semaine'] = pd.to_datetime(df['Date_Debut_x']).dt.dayofweek
    
    # Create weekend indicator
    df['weekend'] = (df['jour_semaine'] >= 5).astype(int)
    
    # Create event indicator
    df['evenement'] = df['Event'].fillna(0).astype(int)
    
    # Select and reorder columns
    processed_df = df[[
        'temperature', 'humidite', 'vitesse_vent', 'precipitation',
        'mois', 'jour_semaine', 'weekend', 'evenement', 'consommation'
    ]]
    
    # Save processed data
    processed_df.to_csv('data/processed/energy_data_processed.csv', index=False)
    print("Data preprocessing complete! Processed data saved to data/processed/energy_data_processed.csv")

if __name__ == "__main__":
    preprocess_data() 