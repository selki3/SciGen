import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler, AutoTokenizer, DataCollatorWithPadding
from tqdm.auto import tqdm
from utils import Table2textDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Instantiate the Table2textDataset
table2text_dataset = Table2textDataset(
    tokenizer,
    data_dir="./data/sciLang/",  # Adjust the data directory path
    type_path="train",             # Specify train, validation, or test
    max_source_length=768,         # Maximum length of the source text
    max_target_length=512          # Maximum length of the target text
)

# Define the data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = None  
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 30
num_training_steps = num_epochs * len(table2text_dataset) // 8  # Adjust batch size as needed

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

train_dataloader = DataLoader(
    table2text_dataset, batch_size=8, collate_fn=data_collator
)

progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

model.eval()

metric = None  

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
