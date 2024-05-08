# Overall
This is my task for comparison of State-of-the-Art NLP models. I compared FLAN-T5-Large, T5-Base, BART-Base and BART-Large. To run my attempt of FLAN-T5-Large, simply navigate to the baseline environment and run flan.py. This should output predicts. Unfortunately due to time constraints and an unsolvable bus error on the HPC I was unable to run Flan at 30 Epochs like the other models.

# Prerequisites
This project was developed on Python 3.8 so to run this project, this would be recommended, but later versions should be fine as well.

## Requirements
Please ensure you have the following libraries

pip install transformers
pip install accelerate -U
pip install protobuf
pip install pytorch-lightning
pip install bert-score
pip install sacrebleu
pip install moverscore
pip install pyemd
pip install sentencepiece
pip install huggingface_hub

## How to Run
Once you have installed the above libraries, navigate to the baselines folder, open a command terminal and type in python flan.py. Then, it will train FLAN-T5. Otherwise, follow the instructions below to replicate the original SciGen. The batch size, learning rate and epochs should be the default. Note that you will have to have a different version of Python to run T5 and BART. 


# SciGen

SciGen is a new dataset for reasoning-aware data-to-text generation from scientific tables.


> **Abstract:** We introduce SciGen, a new challenge dataset consisting of tables from scientific articles and their corresponding descriptions, for the task of reasoning-aware data-to-text generation.
Describing scientific tables goes beyond the surface realization of the table content and requires reasoning over table values.
The unique properties of SciGen are that (1) tables mostly contain numerical values, and (2) the corresponding descriptions require arithmetic reasoning.
SciGen is the first dataset that assesses the arithmetic reasoning capabilities of generation models on complex input structures, such as tables from scientific articles, and thus it opens new avenues for future research in reasoning-aware text generation and evaluation.
The core part of SciGen, including the test data, is annotated by one of the authors of the corresponding articles. Such expert annotations do not scale to large training data sizes.
To tackle this, we propose a pipeline for automatically extracting high-quality table-description pairs from the LaTeX sources of scientific articles.
We study the effectiveness of state-of-the-art data-to-text generation models on SciGen and evaluate the results using common metrics and human evaluation. 
Our results and analyses show that adding high-quality unsupervised training data improves the correctness and reduces the hallucination in generated descriptions, however, the ability of state-of-the-art models is still severely limited on this task.

Contact person: Nafise Sadat Moosavi, moosavi@ukp.informatik.tu-darmstadt.de


> This repository contains the SciGen dataset, the anootations for human evaluation, the annotation extraction pipeline, and the code of the examined models. 

## Project structure

* `dataset` -- this folder contains the few-shot, medium, and large splits of the SciGen datasets
* `human_evaluation` -- this folder contains the annotations of our conducted human evaluations
* `extraction_pipeline` -- this folder contains the code for extracting automatic table-description pairs from LaTeX files of scientific articles
* `baselines` -- this folder contains the code of the examined text-to-text generation models


## Citation
Please use the following citation:

```
@article{moosavi:2021:SciGen,
  author    = {Nafise Sadat Moosavi, Andreas R{\"u}ckl{\'e}, Dan Roth, Iryna Gurevych},
  title     = {Learning to Reason for Text Generation from Scientific Tables},
  journal   = {arXiv preprint arXiv:2104.08296},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.08296}
}
```
