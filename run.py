import sys
sys.path.insert(0, 'imojie')

import numpy as np
import random
import regex as re
import time
import glob
import ipdb
import argparse
import shutil
import sys
import os
from os.path import exists
import params
import data
import math
from model import Model, set_seed
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
from imojie.aggregate.score import rescore
from oie_readers.extraction import Extraction
# necessary to ignore lots of numpy+tensorflow warnings
warnings.filterwarnings('ignore')

has_cuda = torch.cuda.is_available()

def get_logger(mode, hparams):
    log_dir = hparams.save+'/logs/'
    if os.path.exists(log_dir+f'{mode}'):
        mode_logs = list(glob.glob(log_dir+f'/{mode}_*'))
        new_mode_index = len(mode_logs)+1
        print('Moving old log to...')
        print(shutil.move(hparams.save +
                          f'/logs/{mode}', hparams.save+f'/logs/{mode}_{new_mode_index}'))
    logger = TensorBoardLogger(
        save_dir=hparams.save,
        name='logs',
        version=mode+'.part')
    return logger

def get_checkpoint_path(hparams):
    if hparams.checkpoint:
        return [hparams.checkpoint]
    else:
        all_ckpt_paths = glob.glob(hparams.save+'/*.ckpt')
        return all_ckpt_paths


def train(hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences):
    model = Model(hparams, meta_data_vocab)
    logger = get_logger('train', hparams)
    trainer = Trainer(enable_progress_bar=True, num_sanity_val_steps=hparams.num_sanity_val_steps, gpus=hparams.gpus, logger=logger,
                      callbacks=checkpoint_callback, min_epochs=hparams.epochs, max_epochs=hparams.epochs, gradient_clip_val=hparams.gradient_clip_val,
                      track_grad_norm=hparams.track_grad_norm)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    if(exists(hparams.save+f'/logs/train.part')):
        shutil.move(hparams.save+f'/logs/train.part', hparams.save+f'/logs/train')


def resume(hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences):
    checkpoint_paths = get_checkpoint_path(hparams)
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    if has_cuda:
        loaded_hparams_dict = torch.load(checkpoint_path)['hyper_parameters']
    else:
        loaded_hparams_dict = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))['hparams']
    current_hparams_dict = vars(hparams)
    loaded_hparams_dict = data.override_args(
        loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
    loaded_hparams = data.convert_to_namespace(loaded_hparams_dict)

    model = Model(loaded_hparams, meta_data_vocab)

    logger = get_logger('resume', hparams)
    trainer = Trainer(enable_progress_bar=True, num_sanity_val_steps=5, gpus=hparams.gpus, logger=logger, callbacks=checkpoint_callback,
                      min_epochs=hparams.epochs, max_epochs=hparams.epochs, resume_from_checkpoint=checkpoint_path, accumulate_grad_batches=int(hparams.accumulate_grad_batches),
                      gradient_clip_val=hparams.gradient_clip_val,  limit_train_batches=hparams.limit_train_batches, track_grad_norm=hparams.track_grad_norm,
                      val_check_interval=hparams.val_check_interval)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if(exists(hparams.save+f'/logs/resume.part')):
        shutil.move(hparams.save+f'/logs/resume.part',
                    hparams.save+f'/logs/resume')

# We can probably merge predict and test. and removing caching when only testing - VA
def test(hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences, mapping=None, conj_word_mapping=None):
    checkpoint_paths = get_checkpoint_path(hparams)
    if not 'train' in hparams.mode:
        if has_cuda:
            loaded_hparams_dict = torch.load(checkpoint_paths[0])['hparams']
        else:
            loaded_hparams_dict = torch.load(checkpoint_paths[0], map_location=torch.device('cpu'))['hparams']
        current_hparams_dict = vars(hparams)
        loaded_hparams_dict = data.override_args(loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
        loaded_hparams = data.convert_to_namespace(loaded_hparams_dict)
    else:
        loaded_hparams = hparams

    model = Model(loaded_hparams, meta_data_vocab)
    if mapping != None:
        model._metric.mapping = mapping
    if conj_word_mapping != None:
        model._metric.conj_word_mapping = conj_word_mapping

    logger = get_logger('test', hparams)
    test_f = open(hparams.save+'/logs/test.txt', 'w')

    for checkpoint_path in checkpoint_paths:
        trainer = Trainer(logger=logger, gpus=hparams.gpus,resume_from_checkpoint=checkpoint_path)
        trainer.test(model, test_dataloader)
        result = model.results
        test_f.write(f'{checkpoint_path}\t{result}\n')
        test_f.flush()
    test_f.close()
    if(exists(hparams.save+f'/logs/test.part')):
        shutil.move(hparams.save+f'/logs/test.part', hparams.save+f'/logs/test')

def predict(hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences, mapping=None, conj_word_mapping=None):
    if hparams.task == 'conj':
        hparams.checkpoint = hparams.conj_model
    if hparams.task == 'oie':
        hparams.checkpoint = hparams.oie_model

    checkpoint_paths = get_checkpoint_path(hparams)
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    if has_cuda:
        if hparams.task == 'conj':
            loaded_hparams_dict = torch.load(checkpoint_path)['hyper_parameters']
        elif hparams.task == 'oie':
            loaded_hparams_dict = torch.load(checkpoint_path)['hyper_parameters']
    else:
        loaded_hparams_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['hyper_parameters']
    current_hparams_dict = vars(hparams)
    loaded_hparams_dict = data.override_args(loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
    loaded_hparams = data.convert_to_namespace(loaded_hparams_dict)
    model = Model(loaded_hparams, meta_data_vocab)

    if mapping != None:
        model._metric.mapping = mapping
    if conj_word_mapping != None:
        model._metric.conj_word_mapping = conj_word_mapping

    logger = None
    trainer = Trainer(gpus=hparams.gpus, logger=logger, resume_from_checkpoint=checkpoint_path)
    start_time = time.time()
    model.all_sentences = all_sentences
    tested = trainer.test(model, dataloaders=test_dataloader, ckpt_path=checkpoint_path)
    end_time = time.time()
    print(f'Total Time taken = {end_time-start_time} s')
    return model

# this splits sentences acc to conj model
# then extracts triples from split sentences
def splitpredict(hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences):
    mapping, conj_word_mapping = {}, {}
    hparams.write_allennlp = True
    # looks for a file with conjunctions split already, if not, uses the
    # conj model provided as input to predict splittings
    if hparams.split_fp == '':
        hparams.task = 'conj'
        hparams.checkpoint = hparams.conj_model
        hparams.model_str = "bert-base-cased"
        hparams.mode = 'predict'
        model = predict(hparams, None, meta_data_vocab, None, None, test_dataloader, all_sentences)
        conj_predictions = model.all_predictions_conj
        sentences_indices = model.all_sentence_indices_conj
        assert len(conj_predictions) == len(sentences_indices)
        all_conj_words = model.all_conjunct_words_conj

        sentences, orig_sentences = [], []
        for i, sentences_str in enumerate(conj_predictions):
            list_sentences = sentences_str.strip('\n').split('\n')
            conj_words = all_conj_words[i]
            if len(list_sentences) == 1:
                orig_sentences.append(list_sentences[0]+' [unused1] [unused2] [unused3]')
                mapping[list_sentences[0]] = list_sentences[0]
                conj_word_mapping[list_sentences[0]] = conj_words
                sentences.append(list_sentences[0]+' [unused1] [unused2] [unused3]')
            elif len(list_sentences) > 1:
                orig_sentences.append(
                    list_sentences[0]+' [unused1] [unused2] [unused3]')
                conj_word_mapping[list_sentences[0]] = conj_words
                for sent in list_sentences[1:]:
                    mapping[sent] = list_sentences[0]
                    sentences.append(sent+' [unused1] [unused2] [unused3]')
            else:
                assert False
        sentences.append('\n')

        count = 0
        for sentence_indices in sentences_indices:
            if len(sentence_indices) == 0:
                count += 1
            else:
                count += len(sentence_indices)
        assert count == len(sentences) - 1

    else:
        with open(hparams.predict_fp, 'r') as f:
            lines = f.read()
            lines = lines.replace("\\", "")

        sentences = []
        orig_sentences = []
        extra_str = " [unused1] [unused2] [unused3]"
        for line in lines.split('\n\n'):
            if len(line) > 0:
                list_sentences = line.strip().split('\n')
                if len(list_sentences) == 1:
                    mapping[list_sentences[0]] = list_sentences[0]
                    sentences.append(list_sentences[0] + extra_str)
                    orig_sentences.append(list_sentences[0] + extra_str)
                elif len(list_sentences) > 1:
                    orig_sentences.append(list_sentences[0] + extra_str)
                    for sent in list_sentences[1:]:
                        mapping[sent] = list_sentences[0]
                        sentences.append(sent + extra_str)
                else:
                    assert False

    hparams.task = 'oie'
    hparams.checkpoint = hparams.oie_model
    hparams.model_str = 'bert-base-cased'
    _, _, split_test_dataset, meta_data_vocab, _ = data.process_data_new(hparams, sentences)
    split_test_dataloader = DataLoader(split_test_dataset, batch_size=hparams.batch_size, collate_fn=data.pad_data_with_ent, num_workers=1)

    model = predict(hparams, None, meta_data_vocab, None, None, split_test_dataloader,
                    mapping=mapping, conj_word_mapping=conj_word_mapping, all_sentences=all_sentences)

    if 'labels' in hparams.type:
        label_lines = get_labels(hparams, model, sentences, orig_sentences, sentences_indices)
        f = open(hparams.out+'.labels','w',encoding='utf-8')
        f.write('\n'.join(label_lines))
        f.close()
    print("Triples written to "+hparams.out+".oie, Conjunctions written to "+hparams.out+".conj")
    

def get_labels(hparams, model, sentences, orig_sentences, sentences_indices):
    label_dict = {0 : 'NONE', 1 : 'ARG1', 2 : 'REL', 3 : 'ARG2', 4 : 'ARG2', 5 : 'NONE'}
    lines = []
    outputs = model.outputs
    idx1, idx2, idx3 = 0, 0, 0
    count = 0
    prev_original_sentence = ''

    for i in range(0, len(sentences_indices)):
        if len(sentences_indices[i]) == 0:
            sentence = orig_sentences[i].split('[unused1]')[0].strip().split()
            sentences_indices[i].append(list(range(len(sentence))))

        lines.append('\n'+orig_sentences[i].split('[unused1]')[0].strip())
        for j in range(0, len(sentences_indices[i])):
            assert len(sentences_indices[i][j]) == len(outputs[idx1]['meta_data'][idx2].strip().split()), ipdb.set_trace()
            sentence = outputs[idx1]['meta_data'][idx2].strip() + ' [unused1] [unused2] [unused3]'
            assert sentence == sentences[idx3]
            original_sentence = orig_sentences[i]
            predictions = outputs[idx1]['predictions'][idx2]

            all_extractions, all_str_labels, len_exts = [], [], []
            for prediction in predictions:
                if prediction.sum().item() == 0:
                    break

                labels = [0] * len(original_sentence.strip().split())
                prediction = prediction[:len(sentence.split())].tolist()
                for idx, value in enumerate(sorted(sentences_indices[i][j])):
                    labels[value] = prediction[idx]

                labels = labels[:-3]
                if 1 not in prediction and 2 not in prediction:
                    continue

                str_labels = ' '.join([label_dict[x] for x in labels])
                lines.append(str_labels)

            idx3 += 1
            idx2 += 1
            if idx2 == len(outputs[idx1]['meta_data']):
                idx2 = 0
                idx1 += 1

    lines.append('\n')
    return lines

def prepare_test_dataset(hparams, model, sentences, orig_sentences, sentences_indices):

    label_dict = {0 : 'NONE', 1 : 'ARG1', 2 : 'REL', 3 : 'ARG2',
                  4 : 'LOC', 5 : 'TYPE'}

    lines = []

    outputs = model.outputs

    idx1, idx2, idx3 = 0, 0, 0
    count = 0
    for i in range(0, len(sentences_indices)):
        if len(sentences_indices[i]) == 0:
            sentence = orig_sentences[i].split('[unused1]')[0].strip().split()
            sentences_indices[i].append(list(range(len(sentence))))

        for j in range(0, len(sentences_indices[i])):
            try:
                assert len(sentences_indices[i][j]) == len(outputs[idx1]['meta_data'][idx2].strip().split()), ipdb.set_trace()
            except:
                ipdb.set_trace()
            sentence = outputs[idx1]['meta_data'][idx2].strip() + ' [unused1] [unused2] [unused3]'
            assert sentence == sentences[idx3]
            original_sentence = orig_sentences[i]
            predictions = outputs[idx1]['predictions'][idx2]

            all_extractions, all_str_labels, len_exts = [], [], []
            for prediction in predictions:
                if prediction.sum().item() == 0:
                    break

                if hparams.rescoring != 'others':
                    lines.append(original_sentence)

                labels = [0] * len(original_sentence.strip().split())
                prediction = prediction[:len(sentence.split())].tolist()
                for idx, value in enumerate(sorted(sentences_indices[i][j])):
                    labels[value] = prediction[idx]
                labels[-3:] = prediction[-3:]
                str_labels = ' '.join([label_dict[x] for x in labels])
                if hparams.rescoring == 'first':
                    lines.append(str_labels)
                elif hparams.rescoring == 'max':
                    for _ in range(0, 5):
                        lines.append(str_labels)
                elif hparams.rescoring == 'others':
                    all_str_labels.append(str_labels)
                    labels_3 = np.array(labels[:-3])
                    extraction = ' '.join(np.array(original_sentence.split())[np.where(labels_3!=0)])
                    all_extractions.append(extraction)
                    len_exts.append(len(extraction.split()))
                else:
                    assert False

            if hparams.rescoring == 'others':
                for ext_i, extraction in enumerate(all_extractions):
                    other_extractions = ' '.join(all_extractions[:ext_i]+all_extractions[ext_i+1:])
                    other_len_exts = sum(len_exts[:ext_i])+sum(len_exts[ext_i+1:])
                    input = original_sentence + ' ' + other_extractions
                    lines.append(input)
                    output = all_str_labels[ext_i] + ' ' + ' '.join(['NONE'] * other_len_exts)
                    lines.append(output)

            idx3 += 1
            idx2 += 1
            if idx2 == len(outputs[idx1]['meta_data']):
                idx2 = 0
                idx1 += 1

    lines.append('\n')
    return lines


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    if hparams.save != None:
        checkpoint_callback = ModelCheckpoint(filename="{epoch:02d}_{val_acc:.3f}", save_top_k=-1, monitor = "val_acc", mode="max")

    else:
        checkpoint_callback = None

    if hparams.task == 'conj':
        hparams.train_fp = 'data/ptb-train.labels' if hparams.train_fp == None else hparams.train_fp
        hparams.dev_fp = 'data/ptb-dev.labels' if hparams.dev_fp == None else hparams.dev_fp
        hparams.test_fp = 'data/ptb-test.labels' if hparams.test_fp == None else hparams.test_fp
        if hparams.debug:
            hparams.train_fp = 'data/ptb-train.labels' if hparams.train_fp == None else hparams.train_fp
    elif hparams.task == 'oie':
        hparams.train_fp = 'data/ent_exclusivity/openie4_labels' if hparams.train_fp == None else hparams.train_fp
        hparams.dev_fp = 'data/ent_exclusivity/dev.txt' if hparams.dev_fp == None else hparams.dev_fp
        hparams.test_fp = 'data/ent_exclusivity/test.txt' if hparams.test_fp == None else hparams.test_fp
        if hparams.debug:
            hparams.train_fp = hparams.dev_fp = hparams.test_fp = 'data/debug_oie.labels'

    hparams.gradient_clip_val = 5 if hparams.gradient_clip_val == None else float(hparams.gradient_clip_val)


    train_dataset, val_dataset, test_dataset, meta_data_vocab, all_sentences = data.process_data_new(hparams)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                  collate_fn=data.pad_data_with_ent, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_size, collate_fn=data.pad_data_with_ent, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, collate_fn=data.pad_data_with_ent, num_workers=1)


    for process in hparams.mode.split('_'):
        globals()[process](hparams, checkpoint_callback, meta_data_vocab, train_dataloader, val_dataloader, test_dataloader, all_sentences)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = params.add_args(parser)
    hyperparams = parser.parse_args()
    set_seed(hyperparams.seed)
    main(hyperparams)

