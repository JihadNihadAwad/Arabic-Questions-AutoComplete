import torch

class PredictionManager:
    """
    Handles making predictions with the trained model.
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def predict_next_word(self, text, top_k=5):
        """
        Predicts the top-k most likely next words given an input text.
        This method is suitable for models like the AutocompleteANNModel.
        """
        self.model.eval()  # Set the model to evaluation mode.
        with torch.no_grad():
            input_sequence = self.processor.encode_sequence(text).unsqueeze(0).to(self.device)  # Encode and prepare input.
            output, _ = self.model(input_sequence)  # Get model output (predictions and attention weights).
            probabilities = torch.softmax(output, dim=1)  # Convert output to probabilities.

            # Get the top-k predictions.
            top_probs, top_indices = torch.topk(probabilities[0], k=top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                word = self.processor.idx_to_word[idx.item()]
                predictions.append((word, prob.item()))

            return predictions

    def predict_next_word_lstm(self, text, top_k=5):
        """
        Predicts the top-k most likely next words given an input text,
        specifically for LSTM models (like AutocompleteLSTMModel).
        Handles single-word inputs correctly.
        Combines forward and backward LSTM outputs for prediction.
        """
        self.model.eval()
        with torch.no_grad():
            input_sequence = self.processor.encode_sequence(text).unsqueeze(0).to(self.device)
            output, _ = self.model(input_sequence)

            # Handle single-word inputs
            if output.dim() == 2:
                output = output.unsqueeze(1)

            # Combine forward and backward outputs (since they're concatenated)
            last_output = output[:, -1, :]  # Take the entire last output vector

            probabilities = torch.softmax(last_output, dim=1).squeeze(0)

            top_probs, top_indices = torch.topk(probabilities, k=top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                word = self.processor.idx_to_word[idx.item()]
                predictions.append((word, prob.item()))

            return predictions