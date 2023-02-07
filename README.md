# Open Information Extraction with Entity Focused Constraints

This repository contains code for the EACL 2023 Findings paper:

Prajna Upadhyay, Oana Balalau, Ioana Manolescu. _Open Information Extraction with Entity Focused Constraints._

## Installation
```
conda create -n openie_entities python=3.9
conda activate openie_entities
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

## Download resources

Download the model and the datasets. It requires installaing `git lfs`.

```
git lfs install
git clone https://huggingface.co/prajnaupadhyay/openie_with_entities
```

This will download the folder `openie_with_entities` with sub-folders `models` and `datasets` under it.

#### Models
##### COORDINATE_BOUNDARY: `models/coordinate_boundary/conj.ckpt`
##### WARM_UP: `models/warmpup.ckpt`
##### CONSTRAINED: `models/clean_seed_777.ckpt`


#### Datasets
There are 3 training datasets:

##### CLEAN: `datasets/train/clean`
##### MIXED: `datasets/train/mixed`
##### ORIGINAL: `datasets/train/original`

There is 1 test dataset:

##### PUBMED GOLD DATA: `datasets/gold/pubmed.tsv`



## Running the model to get triples

```
python run.py --mode splitpredict --inp <path/to/input_file> --out <path/to/output_file>  --task oie --gpus 1 --oie_model models/constrained/clean_seed_777.ckpt --conj_model models/coordinate_boundary/conj.ckpt --ent_extractor flair --num_extractions 5 --type labels
```

This command returns a `.oie` file, which is the output file containing the triples.

## Retraining the model

### Warm up training
```
python run.py --save models/warm_up --mode train_test --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 24 --optimizer adamW --lr 2e-05 --iterative_layers 2 --train_fp openie_with_entities/datasets/train/clean
```
 

### Constrained training

```
python run.py --save models/constrained_model/ --ent_extractor flair --mode resume --model_str bert-base-cased --task oie --epochs num_epochs --gpus 1 --batch_size 24 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint models/warm_up/warmup.ckpt --constraints posm_hvc_hvr_hve_ent-arg_ent-excl_ent-rel_ent_tog --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3_3_3_3_3 --val_check_interval 0.1 --train_fp openie_data/train/clean
```

## Evaluating the scores

First, we have to convert the predicted file into the format accepted by carb/wire57 metrics. So, run the following:

`python carb/evaluation/prepare.py openie_output_file extracted_file `


where

`openie_output_file` is the .oie file returned by OpenIE

`extracted_file` is the file where we want to store this file in the format required by CaRB or OIE16

Secondly, we have to give extracted_file as input to carb / wire57 metrics. They need a virtual environment to be created. More details are here: https://github.com/dair-iitd/CaRB. 

