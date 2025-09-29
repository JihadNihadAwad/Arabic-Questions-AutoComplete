# main.py
import torch
from torch.utils.data import DataLoader
from arabic_processor import ArabicProcessor
from dataset import AutocompleteDataset, AutocompleteTestDataset
from training_manager import TrainingManager
from prediction_manager import PredictionManager
from constants import *
from utils import load_data, get_latest_checkpoint, check_hyperparameter_changes, rename_old_checkpoints, print_device_info, print_sample_predictions, pause_training, check_compatible_hyperparameter_changes
from hyperparameter_manager import HyperparameterManager
from model_setup import load_model_and_optimizer
import os

def main():
    """
    Main function to run the training and evaluation process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_info(device)

    train_questions = load_data('train-open.json')
    val_questions = load_data('val-open.json')
    test_questions = load_data('test-open.json')

    processor = ArabicProcessor()
    processor.build_vocab(train_questions)
    print(f"--> Vocabulary Size: {processor.vocab_size}")

    train_dataset = AutocompleteDataset(train_questions, processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = AutocompleteTestDataset(val_questions, processor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = AutocompleteTestDataset(test_questions, processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    hyperparameter_manager = HyperparameterManager()
    hyperparameter_manager.print_hyperparameters()

    # --- Checkpoint Handling ---
    latest_checkpoint = get_latest_checkpoint()

    # Check for incompatible hyperparameter changes only if a checkpoint exists
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        if check_hyperparameter_changes(checkpoint):
            rename_old_checkpoints(checkpoint)  # Rename the existing checkpoints folder
            print("Incompatible hyperparameters changed. Starting training from scratch.")
            model, optimizer, epoch_start = load_model_and_optimizer(processor, device)
        else:
            check_compatible_hyperparameter_changes(checkpoint)
            model, optimizer, epoch_start = load_model_and_optimizer(processor, device, checkpoint)
    else:
        # If no checkpoint exists, start training from scratch
        print("No checkpoints found. Starting training from scratch.")
        model, optimizer, epoch_start = load_model_and_optimizer(processor, device)

    # Create the checkpoints folder if it doesn't exist
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.makedirs(CHECKPOINTS_FOLDER)

    # --- Load model, optimizer, and epoch_start based on checkpoint status ---
    if 'model' not in locals():  # Check if model is already defined
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            model, optimizer, epoch_start = load_model_and_optimizer(processor, device, checkpoint)
            print(f"Resuming training from epoch: {epoch_start + 1}")
        else:
            model, optimizer, epoch_start = load_model_and_optimizer(processor, device)
            print("Starting training from scratch.")

    training_manager = TrainingManager(model, train_loader, val_loader, test_loader, device, processor, LEARNING_RATE, WEIGHT_DECAY)
    prediction_manager = PredictionManager(model, processor, device)

    hyperparameter_manager.log_hyperparameters()

    print("Starting training loop...")
    training_generator = training_manager.train_model(num_epochs=NUM_EPOCHS_PER_CYCLE, epoch_start=epoch_start)

    while True:
        try:
            for epoch, avg_train_loss, train_accuracy, train_loss, train_perplexity, val_accuracy, val_loss, val_perplexity, test_accuracy, test_loss, test_perplexity in training_generator:
                print_sample_predictions(prediction_manager, epoch)

                if epoch % NUM_EPOCHS_PER_CYCLE == 0 and epoch != 0:
                    if not pause_training():
                        raise StopIteration
                    training_generator = training_manager.train_model(num_epochs=NUM_EPOCHS_PER_CYCLE, epoch_start=epoch)

        except StopIteration:
            print("Training stopped by user.")
            break

if __name__ == '__main__':
    main()