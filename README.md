# Open Information Extraction with Entity Focused Constraints

This repository contains code for the EACL 2023 Findings paper:

Prajna Upadhyay, Oana Balalau, Ioana Manolescu. _Open Information Extraction with Entity Focused Constraints. EACL 2023 Findings Track._

## Installation
```
conda create -n openie_entities python=3.9
conda activate openie_entities
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

## Download resources

Download the model and the datasets.

```

wget https://huggingface.co/prajnaupadhyay/openie_with_entities/blob/main/clean_seed_777.ckpt

wget https://huggingface.co/datasets/prajnaupadhyay/openie_with_entities/tree/main/openie_data

```
There are 3 training datasets:

##### CLEAN: `openie_data/train/clean`
##### MIXED: `openie_data/train/mixed`
##### ORIGINAL: `openie_data/train/original`

There is 1 test dataset:

##### PUBMED GOLD DATA: `openie_data/gold/pubmed.tsv`



## Running the model

```
python run.py --mode splitpredict --inp <path/to/input_file> --out <path/to/output_file>  --task oie --gpus 1 --oie_model <path/to/oie/model> --conj_model <path/to/conj/model> --ent_extractor flair --num_extractions 5 --type labels

```

## Retraining the model

### Warm up training

 

### Constrained training

```
python run.py --save <path/to/save/constrained/model> --ent_extractor flair --mode resume --model_str bert-base-cased --task oie --epochs num_epochs --gpus 1 --batch_size 24 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint <path/to/warmpup/model> --constraints posm_hvc_hvr_hve_ent-arg_ent-excl_ent-rel_ent_tog --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3_3_3_3_3 --val_check_interval 0.1 --train_fp <path/to/training/data>
```

## Predicting the triples

## Evaluating the scores

First, we have to convert the predicted file into the format accepted by carb/OIE16 metrics. So, run the following:


`python Carb/evaluation/prepare.py openie_output_file extracted_file `


where

`openie_output_file` is the .oie file returned by OpenIE

`extracted_file` is the file where we want to store this file in the format required by CaRB or OIE16

Secondly, we have to give extracted_file as input to carb / oi16 metrics. They need a virtual environment to be created. More details are here: https://github.com/dair-iitd/CaRB. 
The gold data is located under `CaRB/evaluation/gold/test_extra.tsv`

