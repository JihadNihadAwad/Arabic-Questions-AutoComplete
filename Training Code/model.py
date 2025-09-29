import torch
import torch.nn as nn
from constants import EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE, NUM_LAYERS

class AdditiveAttention(nn.Module):
    """
    Implements additive attention.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(2 * hidden_dim, 1)  # Learnable weights for attention

    def forward(self, lstm_output, final_hidden):
        """
        Calculates attention scores and context vector.
        """
        # lstm_output: (batch_size, seq_len, hidden_dim)
        # final_hidden: (batch_size, hidden_dim)

        # Repeat the final hidden state for each time step
        final_hidden_repeated = final_hidden.unsqueeze(1).repeat(1, lstm_output.size(1), 1)

        # Concatenate the LSTM output and the repeated final hidden state
        combined_input = torch.cat((lstm_output, final_hidden_repeated), dim=2)

        # Calculate attention energies
        attn_energies = self.attn_weights(combined_input).squeeze(2)  # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_energies, dim=1)  # (batch_size, seq_len)

        # Calculate the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights

class AutocompleteLSTMModel(nn.Module):
    """
    The neural network model for Arabic text autocompletion.
    Now with optional Bi-directional LSTM and Additive Attention.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=dropout_rate)
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_rate if num_layers > 1 else 0)

        # Attention layer (using additive attention)
        self.attention = AdditiveAttention(hidden_dim * (2 if bidirectional else 1))

        # Output layer
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Defines how input data flows through the model.
        """
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)

        if self.bidirectional:
            # Concatenate the hidden states of the last layer from both directions
            final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            final_hidden = hidden[-1, :, :] # Use the hidden state of the last layer

        # Apply attention
        context_vector, attn_weights = self.attention(lstm_out, final_hidden)

        # Output layer
        predictions = self.fc(context_vector)
        predictions = self.fc_dropout(predictions)
        return predictions, attn_weights  # Return attention weights for analysis
    
class AutocompleteANNModel(nn.Module):
    """
    The neural network model for Arabic text autocompletion.
    A simple feedforward neural network.
    """
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(p=dropout_rate)

        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2): # -2 because we've already added one hidden layer and we'll add the output layer next
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.layers = nn.Sequential(*layers)
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Defines how input data flows through the model.
        """
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        # Average the embeddings across the sequence length to get a single vector per input
        # This is a simple way to handle variable-length sequences for a feedforward network.
        embedded = embedded.mean(dim=1) 
        
        out = self.layers(embedded)
        out = self.fc_dropout(out)

        return out, None  # Return None for attention weights (not applicable here)