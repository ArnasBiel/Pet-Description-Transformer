# Transformer training code
# Trains a decoder-style Transformer for next-token prediction on pet descriptions

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# ---------- Hyperparameters ----------
D_MODEL = 512            # Embedding size
NHEAD = 8                # Number of attention heads
NUM_LAYERS = 4           # Number of Transformer layers
DIM_FEEDFORWARD = 1024   # Dimension of feedforward network
DROPOUT = 0.1
MAX_LEN = 1024           # Maximum input sequence length
LR = 1e-4                # Learning rate
EPOCHS = 20
BATCH_SIZE = 8

DATASET_FILE = "pet_description_dataset.pkl"
TOKENIZER_FILE = "pet_description_tokenizer.json"
MODEL_FILE = "transformer_pet_description_model.pth"


# ---------- Model ----------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Using only the encoder as a self-attention stack (GPT-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        seq_len = src.size(1)

        src = self.embedding(src) + self.positional_encoding[:, :seq_len, :]
        src = src.transpose(0, 1)  # -> (seq_len, batch_size, d_model)

        # Causal mask so the model can only attend to previous tokens
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(src, mask=mask)

        output = output.transpose(0, 1)  # -> (batch_size, seq_len, d_model)
        return self.fc_out(output)        # -> (batch_size, seq_len, vocab_size)


# ---------- Dataset ----------
def dataset_generator():
    """Yield entries one-by-one from the pickle file."""
    with open(DATASET_FILE, "rb") as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass


class PetDataset(Dataset):
    def __init__(self, tokenizer, max_len):
        self.data = list(dataset_generator())
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.data[idx])
        input_ids = encoded.ids[: self.max_len]

        # Pad to max_len
        padding_length = self.max_len - len(input_ids)
        input_ids += [self.pad_id] * padding_length

        return torch.tensor(input_ids)


# ---------- Training ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    vocab_size = tokenizer.get_vocab_size()

    dataset = PetDataset(tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TransformerModel(
        vocab_size, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, MAX_LEN
    ).to(device)

    pad_id = tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Next-token prediction: input is everything except the last token,
            # target is everything except the first token
            output = model(batch[:, :-1])
            loss = criterion(
                output.reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Save a checkpoint each epoch
        torch.save(model.state_dict(), f"transformer_pet_description_model_{epoch}.pth")

    # Save final model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Final model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train()
