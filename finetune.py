import json
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric

checkpoint = "models/adrenaline_multiwoz/epoch56_trloss0.40_gpt2"
# checkpoint = "pierreguillou/gpt2-small-portuguese"
with open("data/ontology.json") as fin:
    tokens = json.load(fin)

tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer.add_special_tokens({'additional_special_tokens': tokens})
tokenizer.save_pretrained("models/tokenizer/")
model.resize_token_embeddings(len(tokenizer))
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
    num_proc=4,
    remove_columns=["id", "text"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return { 'accuracy': acc }

training_args = TrainingArguments(
    "test-clm",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=100,
    report_to="wandb",
    run_name=checkpoint,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["valid"],
    # compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(eval_results['eval_loss']):.2f}")
