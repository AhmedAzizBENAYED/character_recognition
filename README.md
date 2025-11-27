📋 Project Structure
character_recognition/
├── config.py          # Centralized configuration
├── data_loader.py     # Data loading & preprocessing
├── model.py           # MLP architecture
├── train.py           # Training pipeline with callbacks
├── evaluate.py        # Evaluation & metrics
├── predict.py         # Inference on new images
└── main.py            # Main execution orchestrator

🎯 Key Features

Advanced MLP Architecture:

Configurable hidden layers (default: 128, 64 neurons)
Batch normalization for stable training
Dropout (0.2) for regularization
L2 weight regularization
ReLU activation with softmax output


Professional Training Pipeline:

Adam optimizer with learning rate scheduling
Early stopping to prevent overfitting
Model checkpointing (saves best model)
ReduceLROnPlateau for adaptive learning
Progress tracking and logging

Comprehensive Evaluation:

Multiple metrics (accuracy, precision, recall, F1)
Confusion matrix visualization
Classification report
Sample prediction visualization

🚀 How to Use
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick demo (trains on MNIST)
python main.py  # Just run quick_demo()

# 3. Or use command line:
python main.py --mode train
python main.py --mode evaluate
python main.py --mode predict --image digit.png

📊 Expected Performance

Test Accuracy: ~98% on MNIST
Training Time: ~5-10 minutes for 20 epochs
Model Size: ~400KB
