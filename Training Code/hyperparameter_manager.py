# hyperparameter_manager.py
import os
from constants import *

class HyperparameterManager:
    """
    Manages hyperparameters, including printing and logging.
    """
    def __init__(self):
        self.log_file = os.path.join(CHECKPOINTS_FOLDER, "training_log.txt")

    def print_hyperparameters(self):
        """
        Prints the hyperparameters.
        """
        print("-" * 50)
        print("Hyperparameters:")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Learning Rate: {LEARNING_RATE}")
        print(f"  Weight Decay: {WEIGHT_DECAY}")
        print(f"  Embedding Dimension: {EMBEDDING_DIM}")
        print(f"  Hidden Dimension: {HIDDEN_DIM}")
        print(f"  Number of LSTM Layers: {NUM_LAYERS}")  # Added number of layers
        print(f"  Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
        print(f"  Number of Epochs per Cycle: {NUM_EPOCHS_PER_CYCLE}")
        print(f"  Dropout Rate: {DROPOUT_RATE}")
        print("-" * 50)

    def log_hyperparameters(self):
        """
        Logs the hyperparameters to the training log file.
        """
        with open(self.log_file, "a") as f:
            f.write("-" * 50 + "\n")
            f.write("Hyperparameters:\n")
            f.write(f"  Batch Size: {BATCH_SIZE}\n")
            f.write(f"  Learning Rate: {LEARNING_RATE}\n")
            f.write(f"  Weight Decay: {WEIGHT_DECAY}\n")
            f.write(f"  Embedding Dimension: {EMBEDDING_DIM}\n")
            f.write(f"  Hidden Dimension: {HIDDEN_DIM}\n")
            f.write(f"  Number of LSTM Layers: {NUM_LAYERS}\n")  # Added number of layers
            f.write(f"  Max Sequence Length: {MAX_SEQUENCE_LENGTH}\n")
            f.write(f"  Number of Epochs per Cycle: {NUM_EPOCHS_PER_CYCLE}\n")
            f.write(f"  Dropout Rate: {DROPOUT_RATE}\n")
            f.write("-" * 50 + "\n")