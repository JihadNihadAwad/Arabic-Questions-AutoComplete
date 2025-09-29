import torch
import re
from constants import MAX_SEQUENCE_LENGTH

class ArabicProcessor:
    """
    Handles preprocessing of Arabic text and manages the vocabulary.
    """
    def __init__(self, max_sequence_length=MAX_SEQUENCE_LENGTH):
        # Initialize dictionaries to map words to indices and vice-versa.
        # '<PAD>' is a special token for padding, '<UNK>' for unknown words.
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.max_sequence_length = max_sequence_length

    def preprocess_text(self, text):
        """
        Cleans and normalizes Arabic text.
        """
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # Remove diacritics (like vowel marks).
        text = re.sub(r'[إأٱآ]', 'ا', text)  # Normalize different forms of 'alef' to a single form.
        text = re.sub(r'ـ+', '', text)  # Remove tatweel (elongation character).
        #text = re.sub(r'ى', 'ي', text)  # Normalize 'ya' and 'alef maqsura' to 'ya'.
        text = re.sub(r'ة\b', 'ه', text)  # Convert 'ta marbuta' at the end of a word to 'ha'.
        text = re.sub(r'([؟])', r' \1 ', text)  # Add spaces around question marks.
        text = re.sub(r'[^\u0600-\u06FF\u0640\u0621-\u065F\s\u061F]', '', text)  # Remove any non-Arabic characters except question marks.
        #text = re.sub(r'(.)\1+', r'\1', text)  # Reduce repeated characters to a single instance (optional).
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single spaces.
        text = text.strip()  # Remove leading and trailing spaces.
        return text

    def build_vocab(self, texts):
        """
        Creates a vocabulary from a list of texts.
        """
        for text in texts:
            words = self.preprocess_text(text).split()
            for word in words:
                if word not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word

    def encode_sequence(self, text):
        """
        Converts a text sequence into a sequence of numerical indices.
        Reverses the sequence for right-to-left processing.
        """
        words = self.preprocess_text(text).split()

        # Limit the sequence length to 'max_sequence_length'.
        if len(words) > self.max_sequence_length:
            words = words[:self.max_sequence_length]
        else:
            # Pad with '<PAD>' if the sequence is shorter.
            words = words + ['<PAD>'] * (self.max_sequence_length - len(words))

        # Reverse the sequence (because arabic is right-to-left)
        words = words[::-1]

        # Convert words to their corresponding indices, using '<UNK>' for unknown words.
        return torch.tensor([self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                             for word in words], dtype=torch.long)

    @property
    def vocab_size(self):
        """
        Returns the total number of unique words in the vocabulary.
        """
        return len(self.word_to_idx)