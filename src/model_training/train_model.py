import torch
import torch.nn as nn
import ast
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
from emoji_data.emoji_information import EmojiParser

emoji_parser = EmojiParser()
EMOJI_LIST = emoji_parser.emoji_list
EMOJI_TO_INDEX = {emoji: idx for idx, emoji in enumerate(EMOJI_LIST)}
INDEX_TO_EMOJI = {idx: emoji for idx, emoji in enumerate(EMOJI_LIST)}

NUM_EMOJIS = len(EMOJI_LIST)
BATCH_SIZE = 1
MAX_LEN = 64
LEARNING_RATE = 1e-6
EPOCHS = 20
TOP_N_EMOJIS = 5
MODEL_SAVE_PATH = "model"

class EmojiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['emojis'].tolist()
        print("texts:", self.texts)
        print("labels:", self.labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        emojis = self.labels[idx]

        # Multi-hot encoding for emojis
        label_vector = np.zeros(NUM_EMOJIS)
        for emoji in emojis:
            if emoji in EMOJI_TO_INDEX:
                label_vector[EMOJI_TO_INDEX[emoji]] = 1

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(label_vector)
        }

class EmojiPredictor(nn.Module):
    def __init__(self, roberta_model_name, num_emojis):
        super(EmojiPredictor, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_emojis)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device, threshold=0.5):
    model = model.eval()
    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int().cpu().numpy()
            labels = labels.cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels)

    return total_loss / len(data_loader), predictions, true_labels


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")

def load_multiple_datasets(file_paths, tokenizer, max_len):
    datasets = []
    for file_path in file_paths:
        print(f"Loading {file_path}")
        df = pd.read_csv(file_path,encoding='utf-8-sig')
        df['emojis'] = df['emojis'].apply(ast.literal_eval)
        print(df)
        dataset = EmojiDataset(df, tokenizer, max_len)
        datasets.append(dataset)
    return datasets

if __name__ == '__main__':
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Initialize model
    model = EmojiPredictor('roberta-base', NUM_EMOJIS)

    # Load existing model if exists
    if os.path.exists(MODEL_SAVE_PATH):
        load_model(model, MODEL_SAVE_PATH, device)
    else:
        model = model.to(device)

    # Loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # List of dataset files to train on
    dataset_files = []

    # Load datasets
    datasets = load_multiple_datasets(dataset_files, tokenizer, MAX_LEN)

    # Train over each dataset
    for round_num, dataset in enumerate(datasets, 1):
        print(f"\n--- Fine-tuning on dataset {round_num} ---")
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
            val_loss, preds, true_labels = eval_model(model, data_loader, loss_fn, device)

            print(f"Round {round_num}, Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model after each dataset fine-tuning
        save_model(model, MODEL_SAVE_PATH)

    # Example inference on a new text
    model.eval()
    test_text = "Cultural Heritage"
    inputs = tokenizer(test_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    probs = [(idx, p) for idx, p in enumerate(probs)]
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    print(probs)
    probs = probs[:20]
    predicted_emojis = [INDEX_TO_EMOJI[idx] for idx, _ in
                        probs if INDEX_TO_EMOJI[idx] in EMOJI_TO_INDEX]
    print(f"\nInput text: {test_text}")
    print(f"Predicted Emojis: {predicted_emojis}")
    
