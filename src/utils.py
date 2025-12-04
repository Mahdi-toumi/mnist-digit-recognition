#!/usr/bin/env python3
# ============================================
# main.py
# ============================================
"""
Script principal pour entraîner et évaluer le modèle MNIST
"""

import argparse
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import train_mnist_model
from utils import load_and_preprocess_mnist, plot_sample_images, plot_class_distribution


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition Training')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'preprocess', 'visualize', 'all'],
                       help='Mode d\'exécution')
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Nombre d\'époques d\'entraînement')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Taille du batch')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Taux d\'apprentissage')
    
    parser.add_argument('--save_dir', type=str, default='models/',
                       help='Répertoire de sauvegarde')
    
    parser.add_argument('--no_save', action='store_true',
                       help='Ne pas sauvegarder les données preprocessées')
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_arguments()
    
    print("="*60)
    print("🔢 MNIST Digit Recognition System")
    print("="*60)
    
    if args.mode in ['preprocess', 'all']:
        print("\n📊 Preprocessing des données...")
        data = load_and_preprocess_mnist(save=not args.no_save)
        
        if args.mode == 'preprocess':
            print("\n✅ Preprocessing terminé!")
            return
    
    if args.mode in ['visualize', 'all']:
        print("\n📈 Visualisation des données...")
        data = load_and_preprocess_mnist(save=False)
        plot_sample_images(data['X_train'][:10], data['y_train'][:10])
        plot_class_distribution(data['y_train'])
        
        if args.mode == 'visualize':
            print("\n✅ Visualisation terminée!")
            return
    
    if args.mode in ['train', 'all']:
        print("\n🚀 Démarrage de l\'entraînement...")
        
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'save_dir': args.save_dir
        }
        
        try:
            model, history, report = train_mnist_model(config)
            
            print("\n" + "="*60)
            print("🎉 Entraînement terminé avec succès!")
            print(f"   📊 Test Accuracy: {report['test_results']['test_accuracy']:.4f}")
            print(f"   💾 Modèles sauvegardés dans: {args.save_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de l\'entraînement: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()