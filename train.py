import os
import math
import tqdm
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric

files = "clear_threads/*.parquet"
checkpoint = "pierreguillou/gpt2-small-portuguese"

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)

train_files, validation_files = train_test_split(glob.glob(files),
                                                 test_size=0.1)

datasets = load_dataset("parquet", data_files={"train": train_files,
                                               "validation": validation_files})
datasets = datasets.filter(lambda x: x['text'] is not None and x['reply'] is not None)

def tokenize_function(examples):
    result = tokenizer(examples["text"] + examples["reply"],
                       padding="max_length", truncation=True, max_length=64)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = datasets.map(
    tokenize_function,
    num_proc=8,
    batched=True,
    batch_size=128,
    remove_columns=["text", "reply"])

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=2000,
    num_train_epochs=20,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
