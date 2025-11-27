"""
train.py
Training pipeline for the MLP model.
Includes callbacks, early stopping, and history tracking.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)


class Trainer:
    """Handles model training with callbacks and monitoring."""

    def __init__(self, model, data_loader, config):
        """
        Initialize trainer.

        Args:
            model: MLPModel instance
            data_loader: DataLoader instance with loaded data
            config: Configuration object
        """
        self.model = model.get_model()
        self.data_loader = data_loader
        self.config = config
        self.history = None

    def train(self):
        """
        Train the model with configured parameters.

        Returns:
            Training history
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60 + "\n")

        # Setup callbacks
        callbacks = self._setup_callbacks()

        # Train model
        history = self.model.fit(
            self.data_loader.x_train,
            self.data_loader.y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(self.data_loader.x_val, self.data_loader.y_val),
            callbacks=callbacks,
            verbose=1
        )

        self.history = history

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60 + "\n")

        return history

    def _setup_callbacks(self):
        """
        Setup training callbacks.

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Model checkpoint - save best model
        checkpoint = ModelCheckpoint(
            filepath=self.config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        return callbacks

    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training and validation metrics.

        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        history = self.history.history

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # Plot loss
        axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.show()

    def save_history(self, filepath=None):
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save the history
        """
        if filepath is None:
            filepath = self.config.HISTORY_SAVE_PATH

        if self.history is None:
            print("No training history to save.")
            return

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        history_dict = {
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']]
        }

        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)

        print(f"Training history saved to: {filepath}")