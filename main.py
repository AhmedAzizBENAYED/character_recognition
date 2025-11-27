"""
main.py
Main execution script for MLP Character Recognition.
Orchestrates training, evaluation, and prediction.
"""

import os
import argparse
import numpy as np
from tensorflow import keras

from config import Config
from data_loader import DataLoader
from model import MLPModel
from train import Trainer
from evaluate import Evaluator
from predict import Predictor


def main():
    """Main execution function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLP Character Recognition')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'predict'],
                       help='Mode: train, evaluate, or predict')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for prediction')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model')

    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    config.print_config()

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    keras.utils.set_random_seed(config.RANDOM_SEED)

    if args.mode == 'train':
        # TRAINING MODE
        print("\n🚀 Starting Training Mode...")

        # Load data
        data_loader = DataLoader(config)
        data_loader.load_mnist_data()

        # Build and compile model
        mlp_model = MLPModel(config)
        mlp_model.compile_model()
        mlp_model.get_model_summary()

        # Train model
        trainer = Trainer(mlp_model, data_loader, config)
        trainer.train()

        # Save training history
        trainer.save_history()
        trainer.plot_training_history()

        # Evaluate on test set
        evaluator = Evaluator(mlp_model.get_model(), data_loader, config)
        metrics, y_true, y_pred = evaluator.evaluate()
        evaluator.plot_confusion_matrix(y_true, y_pred)
        evaluator.visualize_predictions()

        print("\n✅ Training completed successfully!")

    elif args.mode == 'evaluate':
        # EVALUATION MODE
        print("\n📊 Starting Evaluation Mode...")

        # Load data
        data_loader = DataLoader(config)
        data_loader.load_mnist_data()

        # Load trained model
        model_path = args.model_path if args.model_path else config.MODEL_SAVE_PATH

        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first or provide correct model path.")
            return

        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")

        # Evaluate
        evaluator = Evaluator(model, data_loader, config)
        metrics, y_true, y_pred = evaluator.evaluate()
        evaluator.plot_confusion_matrix(y_true, y_pred)
        evaluator.visualize_predictions()

        print("\n✅ Evaluation completed successfully!")

    elif args.mode == 'predict':
        # PREDICTION MODE
        print("\n🔮 Starting Prediction Mode...")

        if args.image is None:
            print("❌ Please provide an image path using --image argument")
            return

        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return

        # Load trained model
        model_path = args.model_path if args.model_path else config.MODEL_SAVE_PATH

        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first or provide correct model path.")
            return

        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")

        # Predict
        predictor = Predictor(model, config)
        predictor.visualize_prediction(args.image)

        print("\n✅ Prediction completed successfully!")


def quick_demo():
    """
    Quick demo function to train and test the model.
    Can be run directly without command line arguments.
    """
    print("\n" + "="*60)
    print("QUICK DEMO MODE - MLP CHARACTER RECOGNITION")
    print("="*60)

    # Initialize
    config = Config()
    config.EPOCHS = 10  # Reduced for demo

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    keras.utils.set_random_seed(config.RANDOM_SEED)

    # Load data
    print("\n📦 Loading data...")
    data_loader = DataLoader(config)
    data_loader.load_mnist_data()

    # Build model
    print("\n🏗️  Building model...")
    mlp_model = MLPModel(config)
    mlp_model.compile_model()
    mlp_model.get_model_summary()

    # Train
    print("\n🎯 Training model...")
    trainer = Trainer(mlp_model, data_loader, config)
    trainer.train()
    trainer.plot_training_history()

    # Evaluate
    print("\n📊 Evaluating model...")
    evaluator = Evaluator(mlp_model.get_model(), data_loader, config)
    metrics, y_true, y_pred = evaluator.evaluate()
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.visualize_predictions()

    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == '__main__':
    # For command line interface, uncomment:
    main()

    # For quick demo (no arguments needed), uncomment:
    # quick_demo()