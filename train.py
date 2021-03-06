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
BLOCK_SIZE = 128

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)

train_files, validation_files = train_test_split(glob.glob(files)[:2],
                                                 test_size=0.1)

datasets = load_dataset("parquet", data_files={"train": train_files,
                                               "validation": validation_files})
datasets = datasets.filter(lambda x: x['text'] is not None and x['reply'] is\
                           not None and len(x['text']+x['reply']) < 10000)
datasets = datasets.files(lambda x: x['text'].endswith((".", "?", "!")) and\
                          x['reply'].endswith((".", "?", "!")))

special_tokens = ["<sos_u>", "<eos_u>", "<sos_r>", "<eos_r>"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    question =  "<sos_u> "+examples["text"]+" <eos_u>"
    answer = "<sos_r> "+examples["reply"]+" <eos_r>"
    return tokenizer(question+answer)

tokenized_datasets = parsed_datasets.map(
    tokenize_function,
    num_proc=8,
    batched=True,
    remove_columns=["id", "text", "reply"])
    # remove_columns=["id", "text", "reply", "__index_level_0__"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items() }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4)

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=2000,
    num_train_epochs=20,
    report_to="wandb",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
