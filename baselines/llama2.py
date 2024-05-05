import torch 
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForCausalLM, Trainer, GPT2Tokenizer, GPT2Model
import numpy as np
import evaluate
from utils import Table2textDataset as AgendaDataset 
from huggingface_hub import login
from evaluate import multi_list_bleu, get_lines

access_token_read = "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"
access_token_write = "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"
login(token=access_token_read)

# Define tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# Load pretrained model
model = GPT2Model.from_pretrained('gpt2')

def compute_loss(pred):
    return torch.tensor(pred.loss).float().cuda()

# Define Trainer with evaluation metric
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=AgendaDataset(tokenizer=tokenizer, data_dir="./dataset/train/few-shot", type_path="train"),
    eval_dataset=AgendaDataset(tokenizer=tokenizer, data_dir="./dataset/development/few-shot", type_path="validation"),
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_loss,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(results)
