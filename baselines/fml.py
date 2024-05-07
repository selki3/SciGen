import torch 
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, Trainer
import numpy as np
import sacrebleu as scb
from moverscore_v2 import get_idf_dict, word_mover_score
from utils import Table2textFlanDataset


def main():
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    new_tokens = ['[R]', '[C]', '[CAP]']
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="test_trainer", 
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=15,
        logging_strategy="epoch",
    )
    test_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="test", max_source_length=384, max_target_length=384)
    eval_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="dev", max_source_length=384, max_target_length=384)
    preds = create_predictions(model, tokenizer, test_dataset)

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
    
    def get_references(path):
        with open(path, 'r') as target:
            references = [line.strip() for line in target]
        return [[ref] for ref in references]
    
    test_dataset = Table2textFlanDataset(tokenizer, data_dir="../dataset/few-shot", type_path="test", max_source_length=384, max_target_length=384)
    preds = create_predictions(model, tokenizer, test_dataset)
    refs = get_references("../dataset/few-shot/test.target")   
    bleu = scb.corpus_bleu(preds, [refs])
    
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)

    scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, \
                        stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
    moverscore_mean = np.mean(scores)
    moverscore_median = np.median(scores)

    print(f"BLEU Scores: {bleu.score}")
    print(f"MoverScore Mean: {moverscore_mean}, Median: {moverscore_median}")
   
    with open("evaluation_scores.txt", "w") as file:
        file.write(f"BLEU Scores: {bleu.score}\n")
        file.write(f"MoverScore Mean: {moverscore_mean}, Median: {moverscore_median}\n")


if __name__ == "__main__":
    main()
