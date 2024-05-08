import torch 
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSeq2SeqLM, Trainer, GPT2Tokenizer, GPT2Model, AdamW
import numpy as np
import evaluate
from utils import Table2textFlanDataset
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup 
import logging

access_token_read = "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"
access_token_write = "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"
login(token=access_token_read)

def main():
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    new_tokens = ['[R]', '[C]', '[CAP]']
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="train", max_source_length=384, max_target_length=384)
    eval_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="dev", max_source_length=384, max_target_length=384)

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_strategy="epoch",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

    )
    trainer.train()

    def create_predictions(model, tokenizer, data):
        model.eval()
        preds = []
        dataloader = DataLoader(data, batch_size=8)
        for batch in dataloader:
            outputs = model.generate(input_ids=batch['input_ids'].to(model.device), attention_mask=batch['attention_mask'].to(model.device))
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend(decoded_outputs)
        return preds
    
    def get_references():
        with open("../dataset/few-shot/test.target", 'r') as target:
            references = [line.strip() for line in target]
        return [[ref] for ref in references]
    
    test_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="test", max_source_length=384, max_target_length=384)
    preds = create_predictions(model, tokenizer, test_dataset)
    refs = get_references()   
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=preds, references=refs)
    print(f"BLEU: {bleu_score}")


if __name__ == "__main__":
    main()