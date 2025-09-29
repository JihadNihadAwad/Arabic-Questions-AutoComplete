# utils.py
import json
import os
import glob
import torch
from constants import CHECKPOINTS_FOLDER, CHECKPOINT_PATH, INCOMPATIBLE_HYPERPARAMETERS, COMPATIBLE_HYPERPARAMETERS

def load_data(file_path):
    """
    Loads JSON data from a file.
    Returns a list where each item is either a question or an answer,
    maintaining the original order from the JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data:
        result.append(item['question'])
        #result.append(item['answer'])
    return result

def get_latest_checkpoint():
    """
    Gets the path to the latest checkpoint file in the checkpoints folder.
    Returns None if no checkpoints are found.
    """
    # Only consider the current CHECKPOINTS_FOLDER
    list_of_files = glob.glob(os.path.join(CHECKPOINTS_FOLDER, '*.pth'))
    if not list_of_files:
        return None

    latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def check_hyperparameter_changes(checkpoint):
    """
    Checks if incompatible hyperparameters have changed compared to the loaded checkpoint.
    """
    for param_name, current_value in INCOMPATIBLE_HYPERPARAMETERS.items():
        if param_name in checkpoint:
            old_value = checkpoint[param_name]  # Get directly from checkpoint
        else:
            print(f"Warning: Could not find hyperparameter '{param_name}' in checkpoint")
            continue

        if old_value != current_value:
            print(f"Incompatible hyperparameter '{param_name}' changed (old: {old_value}, new: {current_value}).")
            return True  # Incompatible change found

    return False  # No incompatible changes

def check_compatible_hyperparameter_changes(checkpoint):
    """
    Checks if compatible hyperparameters have changed and prints a message.
    """
    for param_name, current_value in COMPATIBLE_HYPERPARAMETERS.items():
        if param_name == "learning_rate":
            old_value = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        elif param_name == "weight_decay":
            old_value = checkpoint['optimizer_state_dict']['param_groups'][0]['weight_decay']
        else:
            continue  # Skip other parameters for now

        if old_value != current_value:
            print(f"Compatible hyperparameter '{param_name}' changed (old: {old_value}, new: {current_value}). Training will resume.")

def rename_old_checkpoints(checkpoint):
    """
    Renames the old checkpoints folder based on the old incompatible hyperparameters and epoch.
    """
    old_checkpoint_epoch = checkpoint['epoch']

    # Build folder name using only incompatible hyperparameters
    old_folder_name = f"{CHECKPOINTS_FOLDER}_old"
    for param_name, current_value in INCOMPATIBLE_HYPERPARAMETERS.items():
        if param_name in checkpoint:
            old_value = checkpoint[param_name] # Get directly from the checkpoint
        else:
            print(f"Warning: Could not find hyperparameter '{param_name}' in checkpoint")
            continue
        old_folder_name += f"_{param_name}_{old_value}"
    old_folder_name += f"_epochs_{old_checkpoint_epoch}"

    # Rename the folder
    try:
        os.rename(CHECKPOINTS_FOLDER, old_folder_name)
        print(f"--> Renamed old checkpoint folder to: {old_folder_name}")
    except FileNotFoundError:
        print(f"--> Error: {CHECKPOINTS_FOLDER} not found. It might have already been renamed or moved.")
    except FileExistsError:
        print(f"--> Error: Could not rename {CHECKPOINTS_FOLDER} to {old_folder_name}. Target folder already exists.")


def print_device_info(device):
    """
    Prints information about the device being used for training (CPU or GPU).
    """
    print("-" * 50)
    if device.type == 'cuda':
        print("CUDA is available. Using GPU:")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
    print("-" * 50)

def print_sample_predictions(prediction_manager, epoch):
    """
    Prints sample predictions using the prediction manager.
    """
    print("Making sample predictions...")
    predictions = prediction_manager.predict_next_word("كيف يمكنني", top_k=5)
    print(f"  Predictions after epoch {epoch}:")
    for word, prob in predictions:
        print(f"    - {word}: {prob:.4f}")

def pause_training():
    """
    Pauses the training process and asks the user whether to continue or quit.
    """
    print("-" * 50)
    print("Pausing for input...")
    user_input = input("  - Type 'c' to continue, 'q' to quit: ")
    print("-" * 50)
    return user_input.lower() == 'c'