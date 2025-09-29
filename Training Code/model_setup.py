import torch
import torch.optim as optim
from model import AutocompleteLSTMModel
from model import AutocompleteANNModel
from constants import LEARNING_RATE, WEIGHT_DECAY, HIDDEN_DIM, EMBEDDING_DIM, NUM_LAYERS

def load_model_and_optimizer(processor, device, checkpoint=None):
    """
    Loads the model and optimizer, either from scratch or from a checkpoint.
    """
    if checkpoint:
        num_layers = checkpoint['num_layers']  # Get NUM_LAYERS from checkpoint
        hidden_dim = checkpoint['hidden_dim']  # Get HIDDEN_DIM from checkpoint
        embedding_dim = checkpoint['embedding_dim'] # Get EMBEDDING_DIM from checkpoint
        model = AutocompleteANNModel(processor.vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        print(f"Resuming training from epoch: {epoch_start}")
    else:
        model = AutocompleteANNModel(processor.vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        epoch_start = 0
        print("Starting training from scratch.")

    return model, optimizer, epoch_start