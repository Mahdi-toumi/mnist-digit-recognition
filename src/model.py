# ============================================
# src/model.py
# ============================================
"""
Définition et compilation du modèle CNN pour MNIST
""" 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json
import os


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, 
                   dropout_rate=0.5, l2_reg=0.001):
    """
    Construit un modèle CNN pour la classification MNIST
    
    Args:
        input_shape: Shape des images d'entrée
        num_classes: Nombre de classes (10 pour MNIST)
        dropout_rate: Taux de dropout
        l2_reg: Régularisation L2
    
    Returns:
        keras.Model: Modèle compilé
    """
    
    model = keras.Sequential([
        # Première couche convolutionnelle
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=input_shape,
            name='conv1'
        ),
        layers.BatchNormalization(name='batch_norm1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        layers.Dropout(dropout_rate, name='dropout1'),
        
        # Deuxième couche convolutionnelle
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='conv2'
        ),
        layers.BatchNormalization(name='batch_norm2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        layers.Dropout(dropout_rate, name='dropout2'),
        
        # Couches fully connected
        layers.Flatten(name='flatten'),
        
        layers.Dense(
            units=128,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense1'
        ),
        layers.BatchNormalization(name='batch_norm3'),
        layers.Dropout(dropout_rate, name='dropout3'),
        
        # Couche de sortie
        layers.Dense(
            units=num_classes,
            activation='softmax',
            name='output'
        )
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec les optimiseurs et métriques
    
    Args:
        model: Modèle Keras à compiler
        learning_rate: Taux d'apprentissage
    
    Returns:
        keras.Model: Modèle compilé
    """
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_model_summary(model, save_path='models/model_architecture.json'):
    """
    Crée un résumé détaillé de l'architecture du modèle
    
    Args:
        model: Modèle Keras
        save_path: Chemin pour sauvegarder le résumé
    
    Returns:
        dict: Résumé de l'architecture
    """
    
    summary = {
        'model_type': 'CNN',
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'total_params': model.count_params(),
        'trainable_params': sum([layer.count_params() for layer in model.trainable_weights]),
        'non_trainable_params': sum([layer.count_params() for layer in model.non_trainable_weights]),
        'num_layers': len(model.layers),
        'layers': []
    }
    
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': layer.output_shape,
            'num_params': layer.count_params(),
            'trainable': layer.trainable
        }
        
        # Informations spécifiques selon le type de couche
        if hasattr(layer, 'filters'):
            layer_info['filters'] = layer.filters
        if hasattr(layer, 'kernel_size'):
            layer_info['kernel_size'] = layer.kernel_size
        if hasattr(layer, 'pool_size'):
            layer_info['pool_size'] = layer.pool_size
        if hasattr(layer, 'units'):
            layer_info['units'] = layer.units
        if hasattr(layer, 'activation'):
            layer_info['activation'] = layer.activation.__name__
        if hasattr(layer, 'rate'):
            layer_info['rate'] = layer.rate
        
        summary['layers'].append(layer_info)
    
    # Sauvegarde du résumé
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Résumé du modèle sauvegardé: {save_path}")
    return summary


def build_and_compile_model(input_shape=(28, 28, 1), num_classes=10, 
                          dropout_rate=0.5, l2_reg=0.001, learning_rate=0.001):
    """
    Fonction utilitaire pour construire et compiler le modèle
    
    Returns:
        keras.Model: Modèle construit et compilé
    """
    print("🏗️  Construction du modèle CNN...")
    model = build_cnn_model(input_shape, num_classes, dropout_rate, l2_reg)
    
    print("⚙️  Compilation du modèle...")
    model = compile_model(model, learning_rate)
    
    return model


if __name__ == "__main__":
    # Test du modèle
    model = build_and_compile_model()
    model.summary()
    
    # Créer le résumé
    summary = create_model_summary(model)
    print(f"\n✅ Modèle créé avec {model.count_params():,} paramètres")