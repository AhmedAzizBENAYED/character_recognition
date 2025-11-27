"""
evaluate.py
Model evaluation and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)


class Evaluator:
    """Handles model evaluation and metrics visualization."""

    def __init__(self, model, data_loader, config):
        """
        Initialize evaluator.

        Args:
            model: Trained Keras model
            data_loader: DataLoader instance with test data
            config: Configuration object
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config

    def evaluate(self):
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL ON TEST SET")
        print("=" * 60 + "\n")

        # Get predictions
        y_pred_proba = self.model.predict(self.data_loader.x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.data_loader.y_test, axis=1)

        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(
            self.data_loader.x_test,
            self.data_loader.y_test,
            verbose=0
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Print results
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred,
                                   target_names=[str(i) for i in range(10)]))

        metrics = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics, y_true, y_pred

    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()

    def visualize_predictions(self, num_samples=10, save_path='predictions.png'):
        """
        Visualize sample predictions.

        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save the plot
        """
        # Get random samples
        indices = np.random.choice(len(self.data_loader.x_test), num_samples, replace=False)

        samples = self.data_loader.x_test[indices]
        true_labels = np.argmax(self.data_loader.y_test[indices], axis=1)

        # Get predictions
        predictions = self.model.predict(samples, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)
        pred_probs = np.max(predictions, axis=1)

        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            # Reshape to 28x28 for display
            img = samples[i].reshape(28, 28)

            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

            color = 'green' if pred_labels[i] == true_labels[i] else 'red'
            title = f"True: {true_labels[i]}, Pred: {pred_labels[i]}\nConf: {pred_probs[i]:.2f}"
            axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to: {save_path}")
        plt.show()