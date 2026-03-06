# Tokenizer training code
# Trains a WordPiece tokenizer on the pet description dataset

import pickle
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

DATASET_FILE = "pet_description_dataset.pkl"
TOKENIZER_FILE = "pet_description_tokenizer.json"

# Special tokens used in training — START/END wrap descriptions, rest are standard BERT tokens
SPECIAL_TOKENS = [
    "<START_DESCRIPTION>",
    "<END_DESCRIPTION>",
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


def load_dataset(path):
    """Read all entries back from the pickle file."""
    text_data = []
    with open(path, "rb") as f:
        try:
            while True:
                text_data.append(pickle.load(f))
        except EOFError:
            pass
    return text_data


def train_tokenizer(text_data):
    """Train a WordPiece tokenizer and save it to disk."""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=3000,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(text_data, trainer)
    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer saved to {TOKENIZER_FILE}")
    return tokenizer


if __name__ == "__main__":
    data = load_dataset(DATASET_FILE)
    tokenizer = train_tokenizer(data)

    # Quick sanity check
    encoded = tokenizer.encode("hello there general kenobi")
    print(f"Encoded: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
