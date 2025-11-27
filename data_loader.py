"""
data_loader.py
Data loading and preprocessing module.
Handles MNIST dataset and custom image preprocessing.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os


class DataLoader:
    """Handles data loading and preprocessing for character recognition."""

    def __init__(self, config):
        """
        Initialize DataLoader with configuration.

        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def load_mnist_data(self):
        """
        Load and preprocess MNIST dataset.

        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
        """
        print("Loading MNIST dataset...")

        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Preprocess images
        x_train = self._preprocess_images(x_train)
        x_test = self._preprocess_images(x_test)

        # Split training data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=y_train
        )

        # One-hot encode labels
        y_train = to_categorical(y_train, self.config.NUM_CLASSES)
        y_val = to_categorical(y_val, self.config.NUM_CLASSES)
        y_test = to_categorical(y_test, self.config.NUM_CLASSES)

        # Store data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        # Print dataset information
        self._print_dataset_info()

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _preprocess_images(self, images):
        """
        Preprocess images for MLP input.

        Args:
            images: numpy array of images

        Returns:
            Preprocessed and flattened images
        """
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0

        # Flatten images for MLP (28x28 -> 784)
        images = images.reshape(images.shape[0], -1)

        return images

    def load_custom_image(self, image_path):
        """
        Load and preprocess a custom image for prediction.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image ready for model input
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Resize to 28x28
        img = img.resize((self.config.IMG_WIDTH, self.config.IMG_HEIGHT))

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize and flatten
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, -1)

        return img_array

    def _print_dataset_info(self):
        """Print information about the loaded dataset."""
        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Validation samples: {self.x_val.shape[0]}")
        print(f"Test samples: {self.x_test.shape[0]}")
        print(f"Input shape: {self.x_train.shape[1:]}")
        print(f"Number of classes: {self.config.NUM_CLASSES}")
        print("=" * 60 + "\n")