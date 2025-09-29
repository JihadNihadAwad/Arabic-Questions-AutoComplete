# Arabic Questions AutoComplete

LSTM & ANN Training Code for An Arabic Question Auto Complete Server

This repository contains code and demos for deep learning models that perform auto-completion for Arabic questions. The models are implemented in Python and Java, with a focus on LSTM, ANN, and Decision Tree architectures.

---

## Features

- **Auto-completion for Arabic questions** using LSTM, ANN, and Tree-based models
- Models served with Flask APIs on different ports
- Training code using PyTorch (v2.3.1+ recommended, CUDA 11.8 supported but optional)
- Includes support for Java (1.8+) for serving the model as a JAR
- Evaluation metrics: accuracy, top-k accuracy, and perplexity

---

## Getting Started

### Requirements

- Python 3.7+
- PyTorch 2.3.1 (or compatible; CUDA 11.8 optional)
- Flask
- scikit-learn (for Tree model)
- Java 1.8+ (for Java server)
- A trained model checkpoint (see [Notes](#notes-on-training--checkpoints))

### Running the Servers

Each model runs on a different port:

- **LSTM model:** `python LSTM_run.py` (serves on port **5002**)
- **ANN model:** `python ANN_run.py` (serves on port **5001**)
- **Tree model:** `python Tree_run.py` (serves on port **5003**)
- **Java server:** `java -jar YourModel.jar`

> At least one model server should be running to use the autocomplete feature.  
> Make sure to provide a model checkpoint file to load the trained weights.

---

## Demo

A simple demo (example usage) is included in the repository. To try it out:

1. Start at least one of the servers as described above.
2. Use the provided client script or send a POST request to the corresponding port with your question.

Example (using `curl` for LSTM model):

```bash
curl -X POST http://localhost:5002/autocomplete -H "Content-Type: application/json" -d '{"question": "متى"}'
```

You should receive a JSON response with the predicted completion(s).

---

## Notes on Training & Checkpoints

- PyTorch version used: **2.3.1** (can run on CPU, but training is much slower)
- Training on GPU takes ~5-30 minutes per epoch depending on model/dataset
- Checkpoints are needed to restore and use the trained models for inference

See `Notes.txt` and `Notes_On_Demo.txt` for details about hyperparameters, experiments, and model evaluation.

---

## Model Evaluation

- **Accuracy:** Measures correct completions
- **Top-k Accuracy:** Checks if the correct completion is among the top-k predictions
- **Perplexity:** Indicates the model's confidence/coherence in its predictions

---
