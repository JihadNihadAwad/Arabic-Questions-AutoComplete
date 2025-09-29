# constants.py

# --- Hyperparameters ---
# These settings control the learning process of the AI model.
BATCH_SIZE = 128  # Batch Size: The number of training examples the model sees before it updates its internal parameters.
                 # A larger batch size can speed up training but requires more memory.
LEARNING_RATE = 0.001  # Learning Rate: Determines the step size the model takes when adjusting its parameters during training.
                        # A smaller learning rate can lead to more precise but slower training.
WEIGHT_DECAY = 1e-7  # Weight Decay (L2 Regularization): A technique to prevent the model from becoming too complex and overfitting the training data.
                    # It adds a penalty to the model's internal parameters, encouraging them to be small.
EMBEDDING_DIM = 128  # Embedding Dimension: The size of the vector used to represent each word. Each word is converted into a vector of this size.
                     # A larger embedding dimension can capture more nuanced word meanings but increases the model's complexity.
HIDDEN_DIM = 128  # Hidden Dimension: The number of units in the LSTM layer, which is responsible for processing sequences of words.
                  # A larger hidden dimension allows the model to learn more complex relationships in the text but increases computational cost.
MAX_SEQUENCE_LENGTH = 10  # Maximum Sequence Length: The maximum number of words considered in each input sequence.
                         # Longer sequences are truncated, and shorter sequences are padded.
NUM_EPOCHS_PER_CYCLE = 50  # Number of Epochs per Cycle: An epoch is one complete pass through the training data. This determines how many epochs are run before pausing training.
DROPOUT_RATE = 0.0 # Dropout Rate: A regularization technique that randomly "drops out" (ignores) a fraction of neurons during each training step.
                  # This helps prevent overfitting.
CHECKPOINT_PATH = 'autocomplete_model.pth'  # Checkpoint Path: The file path where the model's state (its learned parameters) is saved.
CHECKPOINTS_FOLDER = 'checkpoints'  # Checkpoints Folder: The folder where checkpoints (saved model states) are stored.

NUM_LAYERS = 3  # Number of LSTM layers

# Hyperparameters that make checkpoints incompatible if changed (so it creates a new folder):
# Changing these parameters changes the model itself.
INCOMPATIBLE_HYPERPARAMETERS = {
    "embedding_dim": EMBEDDING_DIM,  # Changing the embedding dimension changes the size of the word vectors, making old word representations incompatible.
    "hidden_dim": HIDDEN_DIM,  # Changing the hidden dimension alters the number of units in the LSTM layer, making the old learned parameters incompatible.
    "num_layers": NUM_LAYERS,
}

# Was going to include others here but no need
COMPATIBLE_HYPERPARAMETERS = {
    "learning_rate": LEARNING_RATE,  
    "weight_decay": WEIGHT_DECAY,  
}