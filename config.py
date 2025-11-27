"""
config.py
Configuration file for the MLP Character Recognition project.
Contains all hyperparameters and settings.
"""


class Config:
    """Configuration class for model training and architecture."""

    # Data parameters
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    NUM_CHANNELS = 1  # Grayscale
    NUM_CLASSES = 10  # Digits 0-9 (can be extended to 62 for alphanumeric)

    # Model architecture
    HIDDEN_LAYERS = [128, 64]  # Two hidden layers
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'softmax'
    DROPOUT_RATE = 0.2

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Optimizer
    OPTIMIZER = 'adam'

    # Paths
    MODEL_SAVE_PATH = 'models/mlp_character_recognition.h5'
    HISTORY_SAVE_PATH = 'models/training_history.json'

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @classmethod
    def get_input_shape(cls):
        """Returns the input shape for the model."""
        return (cls.IMG_HEIGHT * cls.IMG_WIDTH,)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 60)
        print("MODEL CONFIGURATION")
        print("=" * 60)
        print(f"Input Shape: {cls.get_input_shape()}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Hidden Layers: {cls.HIDDEN_LAYERS}")
        print(f"Activation: {cls.ACTIVATION}")
        print(f"Dropout Rate: {cls.DROPOUT_RATE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Optimizer: {cls.OPTIMIZER}")
        print("=" * 60)