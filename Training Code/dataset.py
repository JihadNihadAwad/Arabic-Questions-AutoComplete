import torch
from torch.utils.data import Dataset

class AutocompleteDataset(Dataset):
    """
    Prepares the training dataset for the autocomplete model.
    """
    def __init__(self, questions, processor):
        self.samples = []
        self.processor = processor

        for question in questions:
            self.add_samples_from_question(question)

    def add_samples_from_question(self, question):
        """
        Adds input-target pairs from a single question to the dataset.
        """
        words = self.processor.preprocess_text(question).split()
        if len(words) < 2:  # Skip questions that are too short (if any anyway).
            return

        for i in range(len(words) - 1):
            self.add_sample(words, i)
    
    def add_sample(self, words, i):
        """
        Creates and adds a single input-target sample.
        """
        input_sequence = words[max(0, i - self.processor.max_sequence_length + 1):i + 1]
        target_word = words[i + 1]

        input_indices = self.prepare_input_sequence(input_sequence)
        target_idx = self.processor.word_to_idx.get(target_word, self.processor.word_to_idx['<UNK>'])

        self.samples.append((
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(target_idx, dtype=torch.long)
        ))

    def prepare_input_sequence(self, sequence):
        """
        Pads and converts an input sequence to indices.
        """
        if len(sequence) < self.processor.max_sequence_length:
            sequence = ['<PAD>'] * (self.processor.max_sequence_length - len(sequence)) + sequence

        return [self.processor.word_to_idx.get(w, self.processor.word_to_idx['<UNK>']) for w in sequence]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class AutocompleteTestDataset(AutocompleteDataset):
    """
    Prepares the validation and test datasets.
    Inherits from AutocompleteDataset and modifies sample creation.
    """
    def add_sample(self, words, i):
        """
        Creates and adds a single input-target sample for test datasets.
        """
        input_sequence = words[max(0, i - self.processor.max_sequence_length):i]
        target_word = words[i]

        input_indices = self.prepare_input_sequence(input_sequence)
        target_idx = self.processor.word_to_idx.get(target_word, self.processor.word_to_idx['<UNK>'])

        self.samples.append((
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(target_idx, dtype=torch.long)
        ))
    
    def add_samples_from_question(self, question):
        """
        Adds input-target pairs from a single question to the dataset.
        """
        words = self.processor.preprocess_text(question).split()
        if len(words) < 2:
            return

        for i in range(1, len(words)):
            self.add_sample(words, i)