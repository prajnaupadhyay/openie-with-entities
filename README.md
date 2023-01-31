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

## Running the model

## Retraining the model

### Warm up training

On the cluster, this file: `/gpfswork/rech/mpe/ucy98jw/pupadhya/openie6/openie6/openie_train.slurm`  is the slurm file to train openie. We have to do the warmup training first, and then the constrained training. Whether to train on the small, cleaned or original data has to be specified in the file run.py: The respective files for small, cleaned and original data are:

```
data/ent_exclusivity/openie4_labels_smaller : small
data/ent_exclusivity/openie4_labels_cleaned1 : cleaned
data/ent_exclusivity/openie4_labels : original
```
The command for warm-up is:

`python run.py --save models_test/ent_exclusivity/smaller/warmup_oie_model--mode train_test --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 24 --optimizer adamW --lr 2e-05 --iterative_layers 2`

where
`--save models_test/ent_exclusivity/smaller/warmup_oie_model  `is the location to save warmup model. 

### Constrained training

After warmpup, the following command is for constrained training:

`python run.py --save models_test/ent_exclusivity/smaller/oie_model1/oie_model1_ent_rel_increased/10x --mode resume --model_str bert-base-cased --task oie --epochs 48 --gpus 1 --batch_size 50 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint models_test/ent_exclusivity/smaller/oie_model1/oie_model1_ent_rel_increased/10x/epoch=28_eval_acc=0.317.ckpt --constraints posm_hvc_hvr_hve_ent-tog_ent-rel_ent-arg - save_k 3 --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3_3_30_3 --val_check_interval 0.1 `

 where
`--save models_test/ent_exclusivity/smaller/oie_model1/oie_model1_ent_rel_increased/10x`   is the location where we want to save the trained model

`--checkpoint models_test/ent_exclusivity/smaller/oie_model1/oie_model1_ent_rel_increased/10x/epoch=28_eval_acc=0.317.ckpt ` specifies to start from a previous ly saved warmup model

`--constraints posm_hvc_hvr_hve_ent-tog_ent-rel_ent-arg`  specify which constraints we want to force

`--cweights 3_3_3_3_3_30_3`  specifies the weights

## Predicting the triples

After training, you can use the slurm file: /gpfswork/rech/mpe/ucy98jw/pupadhya/openie6/openie6/coordinate_boundary_test.slurm  to check the command for prediction. It goes like this:

`python run.py --mode splitpredict --inp results/PROJECTS_relation-extraction_cois_segmented_1_extra.txt --out results/ent_exclusivity/smaller/conj_model/predictions.txt --rescoring --task oie --gpus 1 --oie_model models_test/ent_exclusivity/smaller/oie_model1/epoch=44_eval_acc=0.230.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 --predict_ent sner/PROJECTS_relation-extraction_cois_segmented_1_extra_ent_tagged.txt`

where

` --inp results/PROJECTS_relation-extraction_cois_segmented_1_extra.txt  `

specifies the input file, and this is the good input file with full stops preceded by a space

`--out results/ent_exclusivity/smaller/conj_model/predictions.txt `  specifies the location where we want to save the file

`--oie_model models_test/ent_exclusivity/smaller/oie_model1/epoch=44_eval_acc=0.230.ckpt `  specifies the openie model to use

`--conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt`   specifies the conj model to use

`--predict_ent sner/PROJECTS_relation-extraction_cois_segmented_1_extra_ent_tagged.txt`  specifies the file storing the entities tagged using sner. 

The rest of the parameters that I have used are the same as in the original openie6 code.

## Evaluating the scores

First, we have to convert the predicted file into the format accepted by Carb / OIE16 metrics. So, run the following:


`python Carb/evaluation/prepare.py openie_output_file extracted_file `


where

`openie_output_file` is the .oie file returned by OpenIE

`extracted_file` is the file where we want to store this file in the format required by CaRB or OIE16

Secondly, we have to give extracted_file as input to carb / oi16 metrics. They need a virtual environment to be created. More details are here: https://github.com/dair-iitd/CaRB. 
The gold data is located under `CaRB/evaluation/gold/test_extra.tsv`

