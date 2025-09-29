# training_manager.py
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from constants import NUM_EPOCHS_PER_CYCLE, CHECKPOINT_PATH, CHECKPOINTS_FOLDER, HIDDEN_DIM, EMBEDDING_DIM, NUM_LAYERS

class TrainingManager:
    """
    Handles the training and evaluation process of the model.
    """
    def __init__(self, model, train_loader, val_loader, test_loader, device, processor, learning_rate, weight_decay):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.processor = processor
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.log_file = os.path.join(CHECKPOINTS_FOLDER, "training_log.txt")
        self.best_val_loss = float('inf')

    def train_model(self, num_epochs=NUM_EPOCHS_PER_CYCLE, epoch_start=0):
        """
        Trains the autocomplete model.
        """
        k = 5  # Local variable to store k
        for epoch in range(epoch_start, epoch_start + num_epochs):
            self.model.train()  # Set the model to training mode.
            total_loss = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # Move data to the device.

                self.optimizer.zero_grad()  # Reset gradients from the previous iteration.
                outputs, _ = self.model(inputs)  # Forward pass: Get model predictions.
                loss = self.criterion(outputs, targets)  # Calculate the loss.

                loss.backward()  # Backward pass: Compute gradients.
                self.optimizer.step()  # Update model parameters.

                total_loss += loss.item()

                # Print training progress every 100 batches.
                if batch_idx % 100 == 0:
                    print(f"\r--> Epoch {epoch+1}/{num_epochs + epoch_start} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}", end="")

            print()
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"--> Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

            # Evaluate the model after each epoch.
            train_accuracy, train_topk_accuracy, train_loss, train_perplexity = self.evaluate_model(self.train_loader, k=k)
            print(f"--> Training - Accuracy: {train_accuracy:.2f}% | Top-{k} Accuracy: {train_topk_accuracy:.2f}% | Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.4f}")

            val_accuracy, val_topk_accuracy, val_loss, val_perplexity = self.evaluate_model(self.val_loader, k=k)
            print(f"--> Validation - Accuracy: {val_accuracy:.2f}% | Top-{k} Accuracy: {val_topk_accuracy:.2f}% | Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.4f}")

            test_accuracy, test_topk_accuracy, test_loss, test_perplexity = self.evaluate_model(self.test_loader, k=k)
            print(f"--> Test - Accuracy: {test_accuracy:.2f}% | Top-{k} Accuracy: {test_topk_accuracy:.2f}% | Loss: {test_loss:.4f} | Perplexity: {test_perplexity:.4f}")

            # Log results every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.log_results(epoch, train_accuracy, train_topk_accuracy, train_loss, train_perplexity,
                                 val_accuracy, val_topk_accuracy, val_loss, val_perplexity,
                                 test_accuracy, test_topk_accuracy, test_loss, test_perplexity, k)

            self.save_checkpoint(epoch)
            self.save_best_model(val_loss, epoch)
            print("-" * 50)

            yield epoch + 1, avg_train_loss, train_accuracy, train_loss, train_perplexity, val_accuracy, val_loss, val_perplexity, test_accuracy, test_loss, test_perplexity

    def evaluate_model(self, data_loader, k=5):
        """
        Evaluates the model's performance on a given dataset.
        """
        self.model.eval()  # Set the model to evaluation mode.
        total_correct = 0
        total_topk_correct = 0  # For top-k accuracy.
        total_samples = 0
        total_loss = 0

        with torch.no_grad():  # Disable gradient calculations during evaluation.
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _ = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Get the most likely prediction.
                total_correct += (predicted == targets).sum().item()

                # Calculate top-k accuracy.
                _, topk_predicted = torch.topk(outputs, k, dim=1)
                total_topk_correct += (topk_predicted == targets.unsqueeze(1)).sum().item()

                total_samples += targets.size(0)

        accuracy = (total_correct / total_samples) * 100
        topk_accuracy = (total_topk_correct / total_samples) * 100
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)  # Perplexity: A measure of how well the model predicts.

        return accuracy, topk_accuracy, avg_loss, perplexity

    def log_results(self, epoch, train_accuracy, train_topk_accuracy, train_loss, train_perplexity,
                    val_accuracy, val_topk_accuracy, val_loss, val_perplexity,
                    test_accuracy, test_topk_accuracy, test_loss, test_perplexity, k):
        """
        Logs the evaluation results to a file.
        """
        log_entry = f"Epoch: {epoch + 1}\n"
        log_entry += f"Training - Accuracy: {train_accuracy:.2f}%, Top-{k} Accuracy: {train_topk_accuracy:.2f}%, Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.4f}\n"
        log_entry += f"Validation - Accuracy: {val_accuracy:.2f}%, Top-{k} Accuracy: {val_topk_accuracy:.2f}%, Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}\n"
        log_entry += f"Test - Accuracy: {test_accuracy:.2f}%, Top-{k} Accuracy: {test_topk_accuracy:.2f}%, Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}\n"
        log_entry += "-" * 50 + "\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)

    def save_checkpoint(self, epoch):
        """
        Saves a checkpoint of the model and optimizer state.
        """
        checkpoint_path = os.path.join(CHECKPOINTS_FOLDER, f'autocomplete_model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'num_layers': NUM_LAYERS, # Save NUM_LAYERS explicitly
            'hidden_dim': HIDDEN_DIM,  # Save HIDDEN_DIM explicitly
            'embedding_dim': EMBEDDING_DIM,  # Save EMBEDDING_DIM explicitly
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'processor_state_dict': {
                'word_to_idx': self.processor.word_to_idx,
                'idx_to_word': self.processor.idx_to_word,
                'max_sequence_length': self.processor.max_sequence_length
            }
        }, checkpoint_path)
        print(f"--> Checkpoint Saved: {checkpoint_path}")

    def save_best_model(self, val_loss, epoch):
        """
        Saves the model if it has the best validation loss so far.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'num_layers': NUM_LAYERS, # Save NUM_LAYERS explicitly
                'hidden_dim': HIDDEN_DIM,  # Save HIDDEN_DIM explicitly
                'embedding_dim': EMBEDDING_DIM,  # Save EMBEDDING_DIM explicitly
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'processor_state_dict': {
                    'word_to_idx': self.processor.word_to_idx,
                    'idx_to_word': self.processor.idx_to_word,
                    'max_sequence_length': self.processor.max_sequence_length
                }
            }, CHECKPOINT_PATH)
            print(f"--> Model Saved - Validation Loss improved to {val_loss:.4f}")