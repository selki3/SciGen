import torch 
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForCausalLM, Trainer, GPT2Tokenizer, GPT2Model, AdamW
import numpy as np
import evaluate
from utils import Table2textDataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup 
import logging



access_token_read = "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"
access_token_write = "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"
login(token=access_token_read)


def main():    
    tokenizer = GPT2Tokenizer.from_pretrained(google/flan-t5-base)
    new_tokens = ['[R]', '[C]', '[CAP]']
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

    model = GPT2Model.from_pretrained(google/flan-t5-base)

    train_dataset = Table2textDataset(tokenizer, data_dir="../dataset/few-shot", type_path="train", max_source_length=384, max_target_length=384)
    eval_dataset = Table2textDataset(tokenizer, data_dir="../dataset/few-shot", type_path="dev", max_source_length=384, max_target_length=384)

    print(train_dataset)
    print(eval_dataset)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()