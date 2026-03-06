# Pet Description Transformer

A Transformer-based language model trained to generate pet adoption descriptions, built on data from PetFinder.  
The project also includes KMeans clustering on linguistic features to analyse and potentially enhance model training.

## Overview

The idea is straightforward: take structured pet attributes (species, colour, age, gender, size) along with their adoption descriptions, and train a next-token-prediction Transformer that can learn to produce similar descriptions. On top of that, linguistic features extracted from the text (tone, emotion, formality, etc.) are clustered with KMeans to find natural groupings in the data. These clusters could later be fed back into the model as conditioning signals.

## Project Structure

```
pet-description-transformer/
├── data_processing.py       # Reads raw CSVs and serialises entries to a pickle dataset
├── train_tokenizer.py       # Trains a WordPiece tokenizer on the dataset
├── tokenize_data.py         # Applies the tokenizer and saves tokenized output as JSON
├── train_model.py           # Defines and trains the Transformer (next-token prediction)
├── clustering.py            # KMeans clustering on linguistic features + PCA visualisation
├── data_exploration.py      # Explores the dataset: volume, variety, velocity, veracity
├── requirements.txt         # Python dependencies
└── README.md
```

## Pipeline

The scripts are meant to be run in order:

### 1. Data Processing
```bash
python data_processing.py
```
Reads `petfinder_study1.csv` and `petfinder_study2.csv`, formats each pet's attributes and description into a structured text entry, and writes them to `pet_description_dataset.pkl`.

### 2. Tokenizer Training
```bash
python train_tokenizer.py
```
Trains a WordPiece tokenizer (vocab size 3000) on the dataset. Special tokens include `<START_DESCRIPTION>` and `<END_DESCRIPTION>` to delimit the description text. Saves to `pet_description_tokenizer.json`.

### 3. Data Tokenization
```bash
python tokenize_data.py
```
Applies the trained tokenizer to every entry and dumps the tokenized sequences into `pet_descriptions_tokenized.json` for inspection.

### 4. Model Training
```bash
python train_model.py
```
Trains a Transformer model on the tokenized pet descriptions using next-token prediction. Checkpoints are saved after each epoch, and the final model goes to `transformer_pet_description_model.pth`.

**Model specs:**
- Embedding size: 512
- Attention heads: 8
- Transformer layers: 4
- Feedforward dimension: 1024
- Max sequence length: 1024
- Trained for 20 epochs with Adam (lr = 1e-4)

### 5. Clustering (optional)
```bash
python clustering.py
```
Combines both CSVs, selects linguistic features (Tone, affect, posemo, negemo, etc.), standardises them, and runs KMeans (k=5). Outputs a PCA scatter plot (`cluster_visualisation.png`).

The cluster labels could be used to:
- Train separate models per cluster (e.g. one for older pets, one for small dogs)
- Add cluster labels as extra input features to condition the Transformer

### 6. Data Exploration (optional)
```bash
python data_exploration.py
```
Prints stats about the combined dataset organised around the 4 V's: Volume, Variety, Velocity, and Veracity.

## Setup

```bash
pip install -r requirements.txt
```

Place `petfinder_study1.csv` and `petfinder_study2.csv` in the project root, then run the pipeline scripts in order.

## Data

The project expects two PetFinder CSV files with columns for pet descriptions, species, colour, age, gender, size, and various linguistic analysis features (LIWC-style). The second file (`study2`) is missing a few columns (`published_date`, `pull_date`, `duration`, `ln_duration`) — these are filled with nulls during loading.

## Notes

- The model uses a causal (autoregressive) mask so it only attends to previous tokens — standard for text generation.
- GPU is used automatically if available, otherwise falls back to CPU.
- The tokenizer and model files are not included in this repo; you need to train them yourself with the scripts above.
