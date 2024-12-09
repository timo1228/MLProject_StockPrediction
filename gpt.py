# Imports
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from data import YahooDataSet
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
import pandas as pd
import math

# Dataset class
class PricePredictionTextDataset(Dataset):
    def __init__(self, X, y, tokenizer, seq_len=30):
        """
        X, y are numpy arrays or DataFrames of features and target.
        seq_len is how many days of data are used as input for prediction.
        """
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Create text samples with variable sequence length
        self.samples = []
        for i in range(len(X) - self.seq_len - 1):
            # Create input sequence: from i to i+seq_len
            seq_features = []
            for j in range(i, i+self.seq_len):
                line = []
                for col_idx, col_name in enumerate(X.columns):
                    val = X.iloc[j, col_idx]
                    line.append(f"{col_name}={val:.4f}")
                seq_features.append(" ".join(line))

            # Target is the close price at i+seq_len
            target_close = y[i+self.seq_len]
            target_str = f"Close={target_close:.4f}"

            # Combine features and target into a single text string
            text = " [SEP] ".join(seq_features) + " [SEP] " + target_str
            tokenized = self.tokenizer(text, truncation=True, max_length=1024)
            self.samples.append(tokenized)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = item["input_ids"]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }

# Data loading
dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()
X_train_df = pd.DataFrame(X_train, columns=["Open","High","Low","Volume","InterestRate","ExchangeRate","VIX","TEDSpread","EFFR","Gold","Oil","Daily_Return","Volatility","Rolling_Mean_Close"])
X_test_df = pd.DataFrame(X_test, columns=["Open","High","Low","Volume","InterestRate","ExchangeRate","VIX","TEDSpread","EFFR","Gold","Oil","Daily_Return","Volatility","Rolling_Mean_Close"])

# Initialize tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if "[SEP]" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]']})

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Allow dynamic seq_len
seq_len = 20  # You can set this to any desired value
train_dataset = PricePredictionTextDataset(X_train_df, y_train, tokenizer, seq_len=seq_len)
test_dataset = PricePredictionTextDataset(X_test_df, y_test, tokenizer, seq_len=seq_len)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt_price_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    save_total_limit=1,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Prediction function
def predict_next_close(X_seq, seq_len=30):
    """
    Given a sequence of recent days X_seq (DataFrame row slices), produce a predicted Close.
    """
    seq_features = []
    for i in range(len(X_seq)):
        line = []
        for col_idx, col_name in enumerate(X_seq.columns):
            val = X_seq.iloc[i, col_idx]
            line.append(f"{col_name}={val:.4f}")
        seq_features.append(" ".join(line))
    input_text = " [SEP] ".join(seq_features) + " [SEP]"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + 10,
            do_sample=False
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Close=" in generated_text:
        last_close_idx = generated_text.rfind("Close=")
        predicted_close_str = generated_text[last_close_idx:].split()[0].replace("Close=", "")
        try:
            predicted_close = float(predicted_close_str)
            return predicted_close
        except:
            return np.nan
    return np.nan

# Evaluation
y_pred = []
for i in range(len(X_test_df)-seq_len-1):
    seq_slice = X_test_df.iloc[i:i+seq_len]
    pred = predict_next_close(seq_slice, seq_len=seq_len)
    y_pred.append(pred)

y_true = y_test[seq_len+1:seq_len+1+len(y_pred)]
y_pred = np.array(y_pred)

mse = np.mean((y_true - y_pred)**2)
print("Test MSE:", mse)
