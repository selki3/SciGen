import torch 
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForCausalLM, Trainer, GPT2Tokenizer, GPT2Model
import numpy as np
import evaluate
from utils import Table2textDataset as AgendaDataset 
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import logging

logger = logging.getLogger(__name__)
class GPT2Trainer:
    def __init__(self, tokenizer, dataset_kwargs, hparams):
        self.tokenizer = tokenizer
        self.dataset_kwargs = dataset_kwargs
        self.hparams = hparams

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = AgendaDataset(self.tokenizer, type_path=type_path, **self.dataset_kwargs)
        logger.info('loading %s dataloader...', type_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=shuffle,
                                num_workers=20)
        logger.info('done')
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size)


access_token_read = "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"
access_token_write = "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"
login(token=access_token_read)

# Define tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
new_tokens = ['[R]', '[C]', '[CAP]']
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

model = GPT2Model.from_pretrained('gpt2')

def compute_loss(pred):
    return torch.tensor(pred.loss).float().cuda()

trainer = GPT2Trainer(
    tokenizer=tokenizer,
    dataset_kwargs={"data_dir": "../dataset/train/few-shot"},
    hparams=training_args,
)

# Train the model
trainer.train_dataloader()

results = trainer.val_dataloader()
print(results)
