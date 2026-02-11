import os
import sys
import time
from datetime import datetime

def clear_screen():
    """Nettoie l'écran de la console"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Affiche l'en-tête de l'application"""
    clear_screen()
    print("=" * 80)
    print("                 SYSTÈME DE PRÉDICTION DE CONSOMMATION ÉNERGÉTIQUE")
    print("=" * 80)
    print()

def print_menu(options):
    """Affiche un menu avec les options données"""
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print()

def wait_for_key():
    """Attend que l'utilisateur appuie sur une touche pour continuer"""
    print("\nAppuyez sur Entrée pour continuer...")
    input()

def run_module(module_name, option=None):
    """Exécute un module Python avec une option facultative"""
    cmd = f"python {module_name}.py"
    if option:
        cmd += f" {option}"
    os.system(cmd)
    wait_for_key()

def data_management_menu():
    """Menu de gestion des données"""
    while True:
        print_header()
        print("GESTION DES DONNÉES")
        print("-" * 80)
        
        options = [
            "Diviser les données (avec normalisation)",
            "Diviser les données (sans normalisation)",
            "Préparer les données pour le modèle",
            "Préparer les données et supprimer la cible des données de test",
            "Retour au menu principal"
        ]
        
        print_menu(options)
        
        choice = input("Choisissez une option (1-5): ")
        
        if choice == "1":
            run_module("data_module", "1")
        elif choice == "2":
            run_module("data_module", "2")
        elif choice == "3":
            run_module("data_module", "3")
        elif choice == "4":
            run_module("data_module", "4")
        elif choice == "5":
            return
        else:
            print("Option invalide. Veuillez réessayer.")
            time.sleep(1)

def model_menu():
    """Menu de gestion des modèles"""
    while True:
        print_header()
        print("GESTION DES MODÈLES")
        print("-" * 80)
        
        options = [
            "Entraîner un modèle",
            "Faire des prédictions",
            "Évaluer le modèle",
            "Retour au menu principal"
        ]
        
        print_menu(options)
        
        choice = input("Choisissez une option (1-4): ")
        
        if choice == "1":
            run_module("model_module", "1")
        elif choice == "2":
            run_module("model_module", "2")
        elif choice == "3":
            run_module("model_module", "3")
        elif choice == "4":
            return
        else:
            print("Option invalide. Veuillez réessayer.")
            time.sleep(1)

def visualization_menu():
    """Menu de visualisation"""
    while True:
        print_header()
        print("VISUALISATION")
        print("-" * 80)
        
        options = [
            "Visualiser les résultats du modèle",
            "Retour au menu principal"
        ]
        
        print_menu(options)
        
        choice = input("Choisissez une option (1-2): ")
        
        if choice == "1":
            run_module("visualization_module", "1")
        elif choice == "2":
            return
        else:
            print("Option invalide. Veuillez réessayer.")
            time.sleep(1)

def main_menu():
    """Menu principal"""
    while True:
        print_header()
        
        options = [
            "Gestion des données",
            "Gestion des modèles",
            "Visualisation",
            "Lancer l'application web",
            "Quitter"
        ]
        
        print_menu(options)
        
        choice = input("Choisissez une option (1-5): ")
        
        if choice == "1":
            data_management_menu()
        elif choice == "2":
            model_menu()
        elif choice == "3":
            visualization_menu()
        elif choice == "4":
            print("Lancement de l'application web...")
            os.system("python app.py")
            wait_for_key()
        elif choice == "5":
            print("Merci d'avoir utilisé le système de prédiction de consommation énergétique.")
            sys.exit(0)
        else:
            print("Option invalide. Veuillez réessayer.")
            time.sleep(1)

if __name__ == "__main__":
    main_menu() 