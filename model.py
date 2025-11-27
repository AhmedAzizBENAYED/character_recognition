"""
model.py
MLP Model architecture for character recognition.
Implements a multi-layer perceptron using Keras Sequential API.
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class MLPModel:
    """Multi-Layer Perceptron model for character recognition."""

    def __init__(self, config):
        """
        Initialize MLP model with configuration.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model = None
        self._build_model()

    def _build_model(self):
        """Build the MLP architecture."""
        print("\nBuilding MLP model...")

        model = Sequential(name='MLP_Character_Recognition')

        # Input layer
        from tensorflow.keras.layers import Input

        model.add(Input(shape=self.config.get_input_shape(), name='input'))
        model.add(Dense(
            self.config.HIDDEN_LAYERS[0],
            activation=self.config.ACTIVATION,
            kernel_regularizer=l2(0.001),
            name='dense_0'
        ))

        model.add(BatchNormalization())
        model.add(Dropout(self.config.DROPOUT_RATE))

        # Hidden layers
        for i, units in enumerate(self.config.HIDDEN_LAYERS[1:], start=1):
            model.add(Dense(
                units,
                activation=self.config.ACTIVATION,
                kernel_regularizer=l2(0.001),
                name=f'hidden_layer_{i}'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config.DROPOUT_RATE))

        # Output layer
        model.add(Dense(
            self.config.NUM_CLASSES,
            activation=self.config.OUTPUT_ACTIVATION,
            name='output_layer'
        ))

        self.model = model
        print("Model built successfully!\n")

    def compile_model(self):
        """Compile the model with optimizer and loss function."""
        optimizer = Adam(learning_rate=self.config.LEARNING_RATE)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model compiled successfully!")

    def get_model_summary(self):
        """Print model architecture summary."""
        print("\n" + "=" * 60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 60)
        self.model.summary()
        print("=" * 60 + "\n")

    def get_model(self):
        """
        Returns the Keras model.

        Returns:
            Compiled Keras model
        """
        return self.model

    def save_model(self, filepath=None):
        """
        Save model to disk.

        Args:
            filepath: Path where to save the model
        """
        if filepath is None:
            filepath = self.config.MODEL_SAVE_PATH

        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath=None):
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model
        """
        if filepath is None:
            filepath = self.config.MODEL_SAVE_PATH

        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")