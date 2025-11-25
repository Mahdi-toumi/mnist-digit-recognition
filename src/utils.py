# ============================================
# src/utils.py
# ============================================
"""
Utilitaires pour le preprocessing des données MNIST
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
import os


def load_and_preprocess_mnist(validation_split=0.2, save=True):
    """
    Charge et preprocess les données MNIST
    
    Args:
        validation_split: Proportion pour validation (défaut: 0.2)
        save: Sauvegarder les données preprocessées (défaut: True)
    
    Returns:
        dict: Dictionnaire contenant toutes les données preprocessées
    """
    
    print("🔄 Chargement des données MNIST...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    print("🔄 Normalisation...")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print("🔄 Reshaping...")
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print("🔄 One-hot encoding...")
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print("🔄 Séparation train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    if save:
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/mnist_preprocessed.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("✅ Données sauvegardées!")
    
    print("✅ Preprocessing terminé!")
    print(f"   Train: {X_train.shape[0]} images")
    print(f"   Val:   {X_val.shape[0]} images")
    print(f"   Test:  {X_test.shape[0]} images")
    
    return data


def load_preprocessed_data(filepath='data/processed/mnist_preprocessed.pkl'):
    """
    Charge les données preprocessées depuis un fichier
    
    Args:
        filepath: Chemin vers le fichier pickle
    
    Returns:
        dict: Données preprocessées
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ Données chargées depuis:", filepath)
    return data


if __name__ == "__main__":
    # Test du script
    data = load_and_preprocess_mnist()
    print("\n Test réussi!")