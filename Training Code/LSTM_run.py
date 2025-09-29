from flask import Flask, request, jsonify
import torch
from arabic_processor import ArabicProcessor
from model import AutocompleteLSTMModel
from prediction_manager import PredictionManager

app = Flask(__name__)

PORT=5002
CHECKPOINT_PATH = 'checkpoints_old_bidirectional_embedding_dim_256_hidden_dim_256_num_layers_3_epochs_127/autocomplete_model_epoch_120.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("-" * 50)
print("Loading saved LSTM model...")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Extract hyperparameters from the checkpoint
embedding_dim = checkpoint['embedding_dim']
hidden_dim = checkpoint['hidden_dim']
num_layers = checkpoint['num_layers']

processor = ArabicProcessor()
processor.word_to_idx = checkpoint['processor_state_dict']['word_to_idx']
processor.idx_to_word = checkpoint['processor_state_dict']['idx_to_word']
processor.max_sequence_length = checkpoint['processor_state_dict']['max_sequence_length']

# Initialise the model using extracted hyperparameters
model = AutocompleteLSTMModel(processor.vocab_size, embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim, num_layers=num_layers).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("...LSTM Model loaded.")
print("-" * 50)

prediction_manager = PredictionManager(model, processor, device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model')
    text = data.get('input')
    num_predictions = data.get('num_predictions', 3)

    if model_type != 'lstm':
        return jsonify({'error': 'Invalid model type for this server'}), 400

    # Flip the input text here (since training data was also flipped)
    text = text.split()
    text.reverse()
    text = " ".join(text)

    predictions = prediction_manager.predict_next_word_lstm(text, top_k=num_predictions)
    output = [{'word': word, 'probability': prob} for word, prob in predictions]

    return jsonify({'model': 'lstm', 'output': output})

if __name__ == '__main__':
    app.run(debug=True, port=PORT)