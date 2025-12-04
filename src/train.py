# ============================================
# src/train.py
# ============================================
"""
Entraînement du modèle CNN pour MNIST
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from .model import build_and_compile_model
from .utils import load_and_preprocess_mnist


class MNISTTrainer:
    """Classe pour gérer l'entraînement du modèle MNIST"""
    
    def __init__(self, config=None):
        """
        Initialise le trainer avec configuration
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.model = None
        self.history = None
        self.data = None
        self.report = {}
        
    def _get_default_config(self):
        """Retourne la configuration par défaut"""
        return {
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'l2_reg': 0.001,
            'early_stopping_patience': 5,
            'reduce_lr_patience': 3,
            'reduce_lr_factor': 0.5,
            'validation_split': 0.2,
            'save_dir': 'models/',
            'save_best_only': True,
            'random_seed': 42
        }
    
    def load_data(self, from_preprocessed=True):
        """Charge les données MNIST"""
        print("📊 Chargement des données...")
        
        if from_preprocessed:
            try:
                from .utils import load_preprocessed_data
                self.data = load_preprocessed_data('data/processed/mnist_preprocessed.pkl')
            except:
                print("⚠️ Fichier preprocessed non trouvé, traitement des données...")
                self.data = load_and_preprocess_mnist(
                    validation_split=self.config['validation_split'],
                    save=True
                )
        else:
            self.data = load_and_preprocess_mnist(
                validation_split=self.config['validation_split'],
                save=True
            )
        
        return self.data
    
    def build_model(self):
        """Construit et compile le modèle"""
        print("🏗️  Construction du modèle...")
        
        self.model = build_and_compile_model(
            input_shape=(28, 28, 1),
            num_classes=10,
            dropout_rate=self.config['dropout_rate'],
            l2_reg=self.config['l2_reg'],
            learning_rate=self.config['learning_rate']
        )
        
        self.model.summary()
        return self.model
    
    def setup_callbacks(self):
        """Configure les callbacks d'entraînement"""
        callbacks = []
        
        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce Learning Rate on Plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['reduce_lr_factor'],
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model Checkpoint
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config['save_dir'], 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=self.config['save_best_only'],
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard (optionnel)
        try:
            log_dir = os.path.join(self.config['save_dir'], 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
            callbacks.append(tensorboard)
        except:
            pass
        
        return callbacks
    
    def train(self, data=None):
        """
        Entraîne le modèle
        
        Args:
            data: Données d'entraînement (optionnel)
        
        Returns:
            dict: Historique d'entraînement
        """
        print("🚀 Début de l'entraînement...")
        
        # Charger les données si non fournies
        if data is None:
            if self.data is None:
                self.load_data()
            data = self.data
        
        # Construire le modèle si non existant
        if self.model is None:
            self.build_model()
        
        # Configurer les callbacks
        callbacks = self.setup_callbacks()
        
        # Entraînement
        start_time = datetime.now()
        
        self.history = self.model.fit(
            data['X_train'],
            data['y_train'],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(data['X_val'], data['y_val']),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"✅ Entraînement terminé en {training_time:.2f} minutes")
        
        # Sauvegarder l'historique
        self._save_training_history()
        
        return self.history
    
    def evaluate(self, data=None):
        """
        Évalue le modèle sur les données de test
        
        Args:
            data: Données de test (optionnel)
        
        Returns:
            dict: Métriques d'évaluation
        """
        print("📈 Évaluation du modèle...")
        
        if data is None:
            if self.data is None:
                self.load_data()
            data = self.data
        
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appelez train() d'abord.")
        
        # Évaluation sur le test set
        test_results = self.model.evaluate(
            data['X_test'],
            data['y_test'],
            verbose=0
        )
        
        # Préparer les résultats
        metric_names = ['test_loss', 'test_accuracy', 'test_precision', 
                       'test_recall', 'test_auc']
        test_metrics = dict(zip(metric_names, test_results))
        
        # Prédictions détaillées
        y_pred_probs = self.model.predict(data['X_test'], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(data['y_test'], axis=1)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Rapport de classification
        class_report = classification_report(y_true, y_pred, 
                                           target_names=[str(i) for i in range(10)],
                                           output_dict=True)
        
        # Statistiques de confiance
        correct_mask = (y_pred == y_true)
        confidence_stats = {
            'correct_predictions': {
                'mean': np.mean(y_pred_probs[correct_mask].max(axis=1)),
                'median': np.median(y_pred_probs[correct_mask].max(axis=1)),
                'min': np.min(y_pred_probs[correct_mask].max(axis=1)),
                'max': np.max(y_pred_probs[correct_mask].max(axis=1))
            },
            'incorrect_predictions': {
                'mean': np.mean(y_pred_probs[~correct_mask].max(axis=1)),
                'median': np.median(y_pred_probs[~correct_mask].max(axis=1)),
                'min': np.min(y_pred_probs[~correct_mask].max(axis=1)),
                'max': np.max(y_pred_probs[~correct_mask].max(axis=1))
            }
        }
        
        # Analyse des paires confondues
        confused_pairs = self._analyze_confusion_pairs(y_true, y_pred, cm)
        
        # Compiler le rapport complet
        self.report = {
            'project_info': {
                'project_name': 'MNIST Handwritten Digit Recognition',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'framework': 'TensorFlow/Keras',
                'dataset': 'MNIST'
            },
            'dataset_info': {
                'total_samples': len(data['X_train']) + len(data['X_val']) + len(data['X_test']),
                'train_samples': len(data['X_train']),
                'validation_samples': len(data['X_val']),
                'test_samples': len(data['X_test']),
                'image_size': '28x28',
                'channels': 1,
                'num_classes': 10
            },
            'model_architecture': self._get_model_summary(),
            'training_config': self.config,
            'training_results': {
                'final_train_accuracy': float(self.history.history['accuracy'][-1]),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'best_val_accuracy': float(np.max(self.history.history['val_accuracy'])),
                'best_val_accuracy_epoch': int(np.argmax(self.history.history['val_accuracy']) + 1),
                'overfitting_gap_percent': float(
                    (self.history.history['accuracy'][-1] - 
                     self.history.history['val_accuracy'][-1]) * 100
                )
            },
            'test_results': {
                'test_accuracy': float(test_metrics['test_accuracy']),
                'test_loss': float(test_metrics['test_loss']),
                'correct_predictions': int(np.sum(correct_mask)),
                'incorrect_predictions': int(np.sum(~correct_mask)),
                'error_rate_percent': float((np.sum(~correct_mask) / len(y_true)) * 100)
            },
            'per_class_metrics': class_report,
            'confusion_analysis': {
                'most_confused_pairs': confused_pairs
            },
            'confidence_statistics': confidence_stats
        }
        
        # Sauvegarder le rapport
        self._save_final_report()
        
        # Visualisations
        self._create_visualizations(data, y_true, y_pred, y_pred_probs, cm)
        
        print(f"✅ Évaluation terminée. Accuracy: {test_metrics['test_accuracy']:.4f}")
        return self.report
    
    def _analyze_confusion_pairs(self, y_true, y_pred, cm):
        """Analyse les paires de digits les plus confondues"""
        confused_pairs = []
        
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    percentage = (cm[i, j] / cm[i].sum()) * 100
                    confused_pairs.append({
                        'true_label': int(i),
                        'predicted_label': int(j),
                        'count': int(cm[i, j]),
                        'percentage': float(percentage)
                    })
        
        # Trier par nombre d'erreurs
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        # Ajouter le rang
        for idx, pair in enumerate(confused_pairs[:10], 1):
            pair['rank'] = idx
        
        return confused_pairs[:10]
    
    def _get_model_summary(self):
        """Récupère le résumé du modèle"""
        from .model import create_model_summary
        return create_model_summary(self.model, 
                                  save_path=os.path.join(self.config['save_dir'], 
                                                       'model_architecture.json'))
    
    def _save_training_history(self):
        """Sauvegarde l'historique d'entraînement"""
        save_path = os.path.join(self.config['save_dir'], 'training_history.pkl')
        
        history_dict = {
            'history': self.history.history,
            'config': self.config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(history_dict, f)
        
        # Sauvegarder aussi en CSV
        history_df = pd.DataFrame(self.history.history)
        csv_path = os.path.join(self.config['save_dir'], 'training_history.csv')
        history_df.to_csv(csv_path, index=False)
        
        print(f"✅ Historique sauvegardé: {save_path}")
    
    def _save_final_report(self):
        """Sauvegarde le rapport final"""
        save_path = os.path.join(self.config['save_dir'], 'final_report.json')
        
        with open(save_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        print(f"✅ Rapport final sauvegardé: {save_path}")
    
    def _create_visualizations(self, data, y_true, y_pred, y_pred_probs, cm):
        """Crée des visualisations des résultats"""
        import matplotlib.pyplot as plt
        
        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Courbes d'apprentissage
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
        plt.close()
        
        # 2. Matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        print(f"✅ Visualisations sauvegardées dans: {save_dir}")
    
    def save_model(self, path=None):
        """Sauvegarde le modèle entraîné"""
        if path is None:
            path = os.path.join(self.config['save_dir'], 'final_model.keras')
        
        self.model.save(path)
        print(f"✅ Modèle sauvegardé: {path}")
        return path


def train_mnist_model(config=None):
    """
    Fonction principale pour entraîner le modèle MNIST
    
    Args:
        config: Configuration d'entraînement
    
    Returns:
        tuple: (model, history, report)
    """
    # Set random seeds
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create trainer
    trainer = MNISTTrainer(config)
    
    # Load data
    trainer.load_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    report = trainer.evaluate()
    
    # Save final model
    trainer.save_model()
    
    return trainer.model, history, report


if __name__ == "__main__":
    # Configuration personnalisée
    custom_config = {
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'save_dir': 'models/',
        'early_stopping_patience': 5
    }
    
    # Entraînement
    model, history, report = train_mnist_model(custom_config)
    
    print("\n" + "="*50)
    print("✅ Entraînement terminé avec succès!")
    print(f"   Test Accuracy: {report['test_results']['test_accuracy']:.4f}")
    print(f"   Modèle sauvegardé dans: models/")
    print("="*50)