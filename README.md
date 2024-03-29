# Open Information Extraction with Entity Focused Constraints

This repository contains code for the EACL 2023 Findings paper:

Prajna Upadhyay, Oana Balalau, Ioana Manolescu. _Open Information Extraction with Entity Focused Constraints._

## Requirements

Python 3.9 is needed to run the project. Please use `git clone` to download the code.

## Install Requirements
The project comes with its `requirements.txt` file which can be installed using the following commands.

```
conda create -n openie_entities python=3.9
conda activate openie_entities
pip install -r requirements.txt
```

## Download Models and Datasets

Download the model and the datasets. It requires installaing `git lfs`.

```
git lfs install
git clone https://huggingface.co/prajnaupadhyay/openie-with-entities
```

This will download the folder `openie-with-entities` with sub-folders `models` and `datasets` under it.

#### Models

##### COORDINATE_BOUNDARY: `models/coordinate_boundary/conj.ckpt`
##### WARM_UP: `models/warmpup.ckpt`
##### CONSTRAINED: `models/clean_seed_777.ckpt`


#### Datasets
There are 3 training datasets:

##### CLEAN: `datasets/train/clean`
##### MIXED: `datasets/train/mixed`
##### ORIGINAL: `datasets/train/original`
##### DEV AND TEST: `datasets/train/dev.txt` and `datasets/train/test.txt`

There is 1 test dataset:

##### PUBMED GOLD DATA: `datasets/gold/pubmed.tsv`


## Running the model to get triples

Please make these changes to the input file before running the model:

i) The file should contain one sentence per line

ii) The coordinate boundary detection model that is used by `openie6` needs all the punctuation symbols such as `,`, `;` or `.` to be preceded by space. For instance, the sentence "Shahrad Taheri received funding from Microsoft, Amazon, and Google.", should be changed to "Shahrad Taheri received funding from Microsoft , Amazon , and Google ." The final results may be affected if this change is not made.

The following command runs the model to generate triples:

```
python run.py --mode splitpredict --inp <path/to/input_file> --out <path/to/output_file>  --task oie --gpus 1 --oie_model openie-with-entities/models/constrained/clean_seed_777.ckpt --conj_model openie-with-entities/models/coordinate_boundary/conj.ckpt --ent_extractor flair --num_extractions 5 --type labels
```

This command returns a `.oie` file, which is the output file containing the triples.

## Retraining the model

### Warm up training
```
python run.py --save models/warm_up --ent_extractor flair --mode train_test --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 24 --optimizer adamW --lr 2e-05 --iterative_layers 2 --train_fp openie-with-entities/datasets/train/clean --dev_fp openie-with-entities/datasets/train/dev.txt --test_fp openie-with-entities/datasets/train/test.txt
```
 

### Constrained training

```
python run.py --save openie-with-entities/models/constrained_model/ --ent_extractor flair --mode resume --model_str bert-base-cased --task oie --epochs num_epochs --gpus 1 --batch_size 24 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint openie-with-entities/models/warm_up/warmup.ckpt --constraints posm_hvc_hvr_hve_ent-arg_ent-excl_ent-rel_ent_tog --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3_3_3_3_3 --val_check_interval 0.1 --train_fp openie-with-entities/datasets/train/clean --dev_fp openie-with-entities/datasets/train/dev.txt --test_fp openie-with-entities/datasets/train/test.txt
```

## Contact
Please contact prajna.u@hyderabad.bits-pilani.ac.in in case of any queries

