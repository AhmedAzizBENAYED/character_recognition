"""
predict.py
Prediction module for inference on new images.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Predictor:
    """Handles predictions on new images."""

    def __init__(self, model, config):
        """
        Initialize predictor.

        Args:
            model: Trained Keras model
            config: Configuration object
        """
        self.model = model
        self.config = config

    def predict_single_image(self, image_path):
        """
        Predict digit from a single image file.

        Args:
            image_path: Path to the image file

        Returns:
            Predicted class and confidence
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img = img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, -1)

        # Predict
        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        print(f"\nPredicted Digit: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")

        # Show all class probabilities
        print("\nClass Probabilities:")
        for i, prob in enumerate(prediction[0]):
            print(f"  {i}: {prob:.4f}")

        return predicted_class, confidence

    def predict_batch(self, image_paths):
        """
        Predict multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        for path in image_paths:
            result = self.predict_single_image(path)
            results.append(result)
        return results

    def visualize_prediction(self, image_path, save_path='prediction_result.png'):
        """
        Visualize prediction with probabilities.

        Args:
            image_path: Path to the image file
            save_path: Path to save the visualization
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('L')
        img_resized = img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array_flat = img_array.reshape(1, -1)

        # Predict
        prediction = self.model.predict(img_array_flat, verbose=0)[0]
        predicted_class = np.argmax(prediction)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Display image
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {prediction[predicted_class]:.4f}',
                     fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Display probabilities
        ax2.bar(range(10), prediction, color='steelblue')
        ax2.set_xlabel('Digit', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(10))
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {save_path}")
        plt.show()