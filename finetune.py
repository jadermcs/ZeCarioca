import math
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# checkpoint = "models/adrenaline_multiwoz/connectcar_tokens"
checkpoint = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
datasets = load_dataset("json", data_files={"train":"data/process.train.json",
                                            "valid":"data/process.valid.json"})
tokenizer.pad_token = tokenizer.eos_token

def add_tokens(examples):
    res = tokenizer(examples['text'], max_length=512, truncation=True,
                    padding='max_length')
    res['labels'] = res['input_ids'].copy()
    return res

tokenized = datasets.map(
    add_tokens,
    batched=True,
    batch_size=32,
    num_proc=4,
    remove_columns=["id", "text"])

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=200,
    num_train_epochs=250,
    report_to="wandb",
    run_name="connectcar-frompierre",
    # run_name="connectcar-fromadrenaline-multiwoz",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["valid"],
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
