import torch
import logging
import sys
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, AutoModelWithLMHead
from tqdm.auto import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import json

class MyTrainer:
    def __init__(self, tokenizer_name=None, cache_dir=None):
        self.base_model = "meta-llama/Llama-2-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else self.base_model, cache_dir=cache_dir)
        new_tokens = ['[R]', '[C]', '[CAP]']
        num_added_toks = self.tokenizer.add_tokens(new_tokens)
        self.model = AutoModelWithLMHead.from_pretrained(self.base_model, num_labels=2)

        logging.info('We have added %s tokens', num_added_toks)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self, train_dataset, eval_datasets, training_args):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_bleu(self, predictions, references):
        return corpus_bleu([[ref] for ref in references], predictions)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {"bleu": self.compute_bleu(predictions, labels)}


# Load train dataset from JSON
train_data_path = "./dataset/train/few-shot/train.json"
with open(train_data_path, "r") as f:
    train_data = json.load(f)

# Load eval datasets from JSON
eval_data_paths = [
    "./dataset/development/few-shot/dev.json",
    "./dataset/test/test-CL.json",
    "./dataset/test/test-Other.json"
]
eval_datasets = []
for eval_data_path in eval_data_paths:
    with open(eval_data_path, "r") as f:
        eval_data = json.load(f)
        eval_datasets.append(eval_data)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = MyTrainer(model_name_or_path, tokenizer_name, cache_dir)
trainer.train(train_data, eval_datasets, training_args)
