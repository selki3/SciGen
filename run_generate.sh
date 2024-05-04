#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=82G
#SBATCH --mail-user=sdillon1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END


module load Anaconda3/2022.10
module load CUDA/10.2.89-GCC-8.3.0

source activate SciGen
cd baselines

python convert_json_files.py -f ../dataset/train/few-shot/train.json -s train
python convert_json_files.py -f ../dataset/development/few-shot/dev.json -s dev
python convert_json_files.py -f ../dataset/test/test-CL.json -s test

mkdir output_train_few_shot_t5_base
mkdir data_few_shots

mv train.* data_few_shots/
mv dev.* data_few_shots/
mv test.* data_few_shots/

python train_table2text_t5.py \
--data_dir=data_few_shots \
--model_name_or_path=google-t5/t5-base \
--learning_rate=3e-5 \
--num_train_epochs 30 \
--train_batch_size=8 \
--eval_batch_size=4 \
--test_batch_size=4 \
--output_dir=output_train_large \
--n_gpu 1 \
--do_train \
--do_predict \
--early_stopping_patience 10 \
--max_source_length 384 \
--max_target_length 384
	   
