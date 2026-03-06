# Data tokenization
# Applies the trained tokenizer to every entry and saves the result as JSON

import json
import pickle
from tokenizers import Tokenizer

DATASET_FILE = "pet_description_dataset.pkl"
TOKENIZER_FILE = "pet_description_tokenizer.json"
OUTPUT_FILE = "pet_descriptions_tokenized.json"


def tokenize_dataset():
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    tokenized_data = []
    with open(DATASET_FILE, "rb") as f:
        try:
            while True:
                text = pickle.load(f)
                encoded = tokenizer.encode(text)
                tokenized_data.append(encoded.tokens)
        except EOFError:
            pass

    with open(OUTPUT_FILE, "w") as out:
        json.dump(tokenized_data, out)

    print(f"Tokenized {len(tokenized_data)} entries → {OUTPUT_FILE}")

    # Print a few examples for testing
    for sentence in tokenized_data[:5]:
        print(sentence)


if __name__ == "__main__":
    tokenize_dataset()
