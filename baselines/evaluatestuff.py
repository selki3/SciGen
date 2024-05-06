import numpy as np
from nltk.translate import meteor_score
from argparse import ArgumentParser
import os
import sacrebleu as scb
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk


def compute_meteor(predictions, references, alpha=0.9, beta=3, gamma=0.5):
    scores = [meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                  for ref, pred in zip(references, predictions)]

    return {"meteor": np.mean(scores)}

def get_lines(fil):
    with open(fil, 'r') as f:
        data = f.read()
    lines = [word_tokenize(sent) for sent in sent_tokenize(data)]
    return lines


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p", "--pred", help="prediction file", required=True)
    parser.add_argument("-s", "--sys", help="system file", required=True)
    parser.add_argument("-all", "--all", help="system file")

    args = parser.parse_args()
    refs = get_lines(args.sys)
    preds = get_lines(args.pred)

    print('Meteor score :', compute_meteor(preds, refs))
    cmd = 'bert-score -r '+args.sys +' -c ' + args.pred + ' --lang en'
    os.system(cmd)
    preds_tokenized = word_tokenize(preds)
    refs_tokenized = word_tokenize(refs)

    if args.all:
        bleu = scb.corpus_bleu([preds_tokenized], [[refs_tokenized]])
        print('BLEU: ', bleu.score)

        idf_dict_hyp = get_idf_dict(preds_tokenized)
        idf_dict_ref = get_idf_dict(refs_tokenized)

        scores = word_mover_score([refs_tokenized], [preds_tokenized], idf_dict_ref, idf_dict_hyp, \
                                  stop_words=[], n_gram=1, remove_subwords=True, batch_size=64)
        print('MoverScre mean: ', np.mean(scores), 'MoverScoreMedian: ', np.median(scores))

