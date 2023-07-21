# Utilities for getting data
import os
import ipdb
import pdb
import random
import argparse
import spacy
import pickle
import csv
import nltk
import numpy as np
from tqdm import tqdm
import re
import torch
from torchtext.legacy import data
from transformers import AutoTokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import numericalize_tokens_from_iterator
import networkx as nx
import sys

import flair
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner-fast')

# tag entities using flair ner



def ent_tags_flair(sentence):
    #print(sentence)
    #print("this is the entity list: "+str(ent_list))

    pos = 0

    s = Sentence(sentence, use_tokenizer = False)

    ent = [0]*len(s)
    reverse_ent= [0]*len(s)
    ent_indices, ent_words=[], []
    ent_pos=[0]*len(s)

    tagger.predict(s)
    entities = s.get_spans('ner')

    #print(entities)
    #for ent_index, entity in enumerate(ent_list):
    for e in entities:
        entity = []
        entity.append(e.text)
        entity.append(e.start_position)
        pos=pos+1
        #print("this entity is: "+str(e.text))
        #print("offset of this entity is: "+str(e.start_position))
        #print("length of sentence = "+str(len(ent)))
        index1 = char_index_to_word_index(s, entity[1], "flair")
        index2 = index1+len(entity[0].split())
        #print("index1 = "+str(index1))
        #print("index2 = "+str(index2))
        if ent_pos[index1] > 0:# we have seen this token in another entity, we skip (Texas-Louisiana problem)
            pos -=1
        else:
            for k in range(index1,index2):
                ent[k]=1
                ent_pos[k]=pos
                ent_indices.append(k)
                ent_words.append(s[k].text.lower())
    for i in range(len(ent)):
        if(ent[i]==0):
            reverse_ent[i]=1
        else:
            reverse_ent[i]=0
    ent.append(0)
    ent.append(0)
    ent.append(0)
    reverse_ent.append(1)
    reverse_ent.append(1)
    reverse_ent.append(1)

    return ent, reverse_ent, ent_indices, ent_words, ent_pos
    #return ent, ent_indices, ent_words




sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)


def remerge_sent(sent):
    # merges tokens which are not separated by white-space
    # does this recursively until no further changes
    # this ensures spacy tokenization does not
    changed = True
    while changed:
        #print(sent)
        #print(changed)
        #print("sent is: "+str(sent)+", len(sent) is: "+str(sent.__len__()))
        changed = False
        i = 0
        while i < sent.__len__() - 1:
            #print("\n i is: "+str(i))
            tok = sent[i]
            #print("tok is: "+str(tok))
            #print(not tok.whitespace_)
            if not tok.whitespace_:
                ntok = sent[i + 1]
                #print("\ntok: "+str(tok)+", ntok: "+str(ntok))
                #print("\ntok.idx: "+str(tok.idx)+", ntok.idx: "+str(ntok.idx))
                #print("len(ntok): "+str(len(ntok)));
                # in-place operation.
                # sent.merge(tok.idx, ntok.idx + len(ntok))
                with sent.retokenize() as retokenizer:
                    retokenizer.merge(sent[i: i+2])
                    #print("length after retokenization: "+str(sent.__len__()))
                changed = True
            i += 1
    return sent


def pos_tags(spacy_sentence):
    pos, pos_indices, pos_words = [], [], []
    for token_index, token in enumerate(spacy_sentence):
        if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
            pos.append(1)
            pos_indices.append(token_index)
            pos_words.append(token.lower_)
        else:
            pos.append(0)
    pos.append(0)
    pos.append(0)
    pos.append(0)
    return pos, pos_indices, pos_words


light_verbs = ["take", "have", "give", "do", "make", "has", "have", "be", "is", "were", "are", "was", "had", "being",
               "began", "am", "following", "having", "do", "does", "did", "started", "been", "became", "left", "help",
               "helped", "get", "keep", "think", "got", "gets", "include", "suggest", "used", "see", "consider",
               "means", "try", "start", "included", "lets", "say", "continued", "go", "includes", "becomes", "begins",
               "keeps", "begin", "starts", "said", "stop", "begin", "start", "continue", "say"]


def verb_tags(spacy_sentence):
    verb, verb_indices, verb_words = [], [], []
    for token_index, token in enumerate(spacy_sentence):
        if token.pos_ in ['VERB'] and token.lower_ not in light_verbs:
            verb.append(1)
            verb_indices.append(token_index)
            verb_words.append(token.lower_)
        else:
            verb.append(0)
    verb.append(0)
    verb.append(0)
    verb.append(0)
    return verb, verb_indices, verb_words


# for each sentence, read the tagged entities and create ds
def ent_tags(sentence, ent_list):
    #print("this is the entity list: "+str(ent_list))
    tokens = sentence.split()
    #print("tokens = "+str(tokens))
    ent = [0] * len(tokens)
    reverse_ent = [0] * len(tokens)
    ent_indices, ent_words = [], []
    ent_pos = [0] * len(tokens)
    pos = 0
    for ent_index, entity in enumerate(ent_list):
        pos = pos + 1
        #print("entity is: "+str(entity))
        #print("offset of this entity is: "+str(entity[1]))
        #print("length of sentence = "+str(len(ent)))
        index1 = char_index_to_word_index(tokens, entity[1], "spacy")
        index2 = index1 + len(entity[0].split(' '))
        #print("index1 = "+str(index1))
        #print("index2 = "+str(index2))
        if ent_pos[index1] > 0:  # we have seen this token in another entity, we skip (Texas-Louisiana problem)
            pos -= 1
        else:
            for k in range(index1, index2):
                ent[k] = 1
                ent_pos[k] = pos
                ent_indices.append(k)
                ent_words.append(tokens[k].lower())
    for i in range(len(ent)):
        if ent[i] == 0:
            reverse_ent[i] = 1
        else:
            reverse_ent[i] = 0
    ent.append(0)
    ent.append(0)
    ent.append(0)
    reverse_ent.append(1)
    reverse_ent.append(1)
    reverse_ent.append(1)

    return ent, reverse_ent, ent_indices, ent_words, ent_pos
    # return ent, ent_indices, ent_words

# given the index of a character in a sentence, it returns the word index
def char_index_to_word_index(sentence, offset, ent_ex):
    tokens = sentence
    #print(tokens)
    #print(len(tokens))
    charIndex = 0
    index = 0
    for i in range(0, len(tokens)):
        if (offset > charIndex):
            # print(str(i)+", "+str(len(tokens[i])))
            if(ent_ex=="flair"):
                charIndex = charIndex + len(tokens[i].text) + 1
            elif(ent_ex=="spacy"):
                charIndex = charIndex + len(tokens[i]) + 1
        else:
            index = i
            break

    return index


# constructs the dependency graph of the sentence and stores it as a networkx graph
def construct_dependency_graph(spacy_sentence):
    edges = []
    count = 0
    dict_num_to_tag = {}
    dict_num_to_word = {}
    dict_min_max_verb = {}
    dict_children = {}
    dict_dep = {}
    for token in spacy_sentence:
        # master=master+" "+token.text
        # print(token)
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop, token.is_sent_start, token.i)
        dict_num_to_word[token.i] = token.text
        dict_num_to_tag[token.i] = token.pos_
        for child in token.children:
            # print(child.tag_)
            # print(token.text+" "+child.text+" "+child.dep_)
            if child.text != ".":
                dict_num_to_word[child.i] = child.text
                dict_num_to_tag[child.i] = child.pos_
                edges.append((token.i, child.i))
                dict_dep[child.i] = child.dep_
    graph = nx.DiGraph(edges)
    return graph, dict_num_to_tag, dict_num_to_word, dict_dep


# return the indices of words that are compound words from the subject word

def find_compound_words(c, graph, dict_dep, list_of_children):
    children_of_children = graph.successors(c)
    for cc in children_of_children:
        if dict_dep[cc] == "compound":
            list_of_children[cc] = ""
            find_compound_words(cc, graph, dict_dep, list_of_children)


# adds the features extracted from the dependency graph
# if the word is a subject or connected to the word that is a subject
def dependency_tags(spacy_sentence):
    dependency, dependency_indices, dependency_words = [], [], []
    dict_dep = {}
    graph, dict_num_to_tag, dict_num_to_word, dict_dep = construct_dependency_graph(spacy_sentence)
    for token_index, token in enumerate(spacy_sentence):
        for child in token.children:
            dict_dep[child.i] = child.dep_
    list_of_children = {}
    for token_index, token in enumerate(spacy_sentence):
        if token_index in dict_dep:
            if dict_dep[token_index] == 'nsubj':
                list_of_children[token_index] = ""
                find_compound_words(token_index, graph, dict_dep, list_of_children)
    for token_index, token in enumerate(spacy_sentence):
        if token_index in list_of_children:
            dependency.append(1)
            dependency_indices.append(token_index)
            dependency_words.append(token.lower_)
        else:
            dependency.append(0)

    dependency.append(0)
    dependency.append(0)
    dependency.append(0)
    return dependency, dependency_indices, dependency_words

# function that creates features from the sentences and the entities tagged in them
def _process_data_new(inp_fp, hparams, fields, tokenizer, label_dict, spacy_model):
    model_str = hparams.model_str
    examples, exampleDs, targets, lang_targets, orig_sentences = [], [], [], [], []
    # print("in _process_data_new")
    # print(inp_fp)
    sentence = None
    max_extraction_length = 5

    if type(inp_fp) == type([]):
        inp_lines = inp_fp
    else:
        inp_lines = open(inp_fp, 'r', encoding="utf-8").readlines()

    new_example = True
    # this for loop constructs the tokens input_ids for each sentences
    for line_num, line in tqdm(enumerate(inp_lines)):
        line = line.strip()
        if line == '':
            new_example = True

        if '[unused' in line or new_example:
            if sentence is not None:
                if len(targets) == 0:
                    targets = [[0]]
                    lang_targets = [[0]]
                orig_sentence = sentence.split('[unused1]')[0].strip()
                orig_sentences.append(orig_sentence)

                exampleD = {'text': input_ids, 'labels': targets[:max_extraction_length], 'word_starts': word_starts,
                            'meta_data': orig_sentence}
                # print("input ids = "+str(input_ids))
                if len(sentence.split()) <= 100 and len(input_ids) < 512:
                    exampleDs.append(exampleD)

                targets = []
                sentence = None
            # starting new example
            if line != '':
                new_example = False
                sentence = line
                length_sen = len(sentence.split())
                tokenized_words = tokenizer.batch_encode_plus(sentence.split(), max_length=512)
                # print("sentence is: "+sentence)
                # print("tokenized words = "+str(tokenized_words))
                # print("hparams.bos_token_id = "+str(hparams.bos_token_id))
                input_ids, word_starts, lang = [hparams.bos_token_id], [], []
                for tokens in tokenized_words['input_ids']:
                    if len(tokens) == 0:  # special tokens like \x9c
                        tokens = [100]
                    word_starts.append(len(input_ids))
                    input_ids.extend(tokens)
                input_ids.append(hparams.eos_token_id)

               # if len(input_ids) >= 512:
                    #print("sentence is: " + sentence + ", number of tokens in the sentence: " + str(len(input_ids)))
                    #count_greater_512 = count_greater_512 + 1

                # print("sentence is: "+sentence+", length of input_ids is: "+len(input_ids))
                assert len(sentence.split()) == len(word_starts), ipdb.set_trace()
        else:
            if sentence is not None:
                target = [label_dict[i] for i in line.split()]
                target = target[:len(word_starts)]
                assert len(target) == len(word_starts), ipdb.set_trace()
                targets.append(target)


    if spacy_model != None:
        sentences = [ed['meta_data'] for ed in exampleDs]
        for sentence_index, spacy_sentence in tqdm(enumerate(spacy_model.pipe(sentences, batch_size=10000))):
            spacy_sentence = remerge_sent(spacy_sentence)
            assert len(sentences[sentence_index].split()) == len(spacy_sentence), ipdb.set_trace()


            pos, pos_indices, pos_words = pos_tags(spacy_sentence)

            dependency, dependency_indices, dependency_words = dependency_tags(spacy_sentence)

            verb, verb_indices, verb_words = verb_tags(spacy_sentence)

            #if (sentences[sentence_index] in sentences_ent_tagged):
            # ent, ent_indices, ent_words = ent_tags(sentences[sentence_index],sentences_ent_tagged[sentences[sentence_index]])
            #ent_list = spacy_model(sentences[sentence_index])
            ent_list_pos = []
            for e in spacy_sentence.ents:
                ent_list_pos.append((e.text, e.start_char, len(e.text)))
            if(hparams.ent_extractor=="spacy"):
                ent, reverse_ent, ent_indices, ent_words, ent_pos = ent_tags(sentences[sentence_index], ent_list_pos)
            elif(hparams.ent_extractor=="flair"):
                ent, reverse_ent, ent_indices, ent_words, ent_pos = ent_tags_flair(sentences[sentence_index])
            #print(sentences[sentence_index])
            #print("ent is: "+str(ent)+", length is: "+str(len(ent)))
            #print("pos is: "+str(pos)+", length is: "+str(len(pos)))

        # check if length of sentence acc to flair tokens is unequal to length of words in spacy tokens
            if(hparams.task == "oie"):
                assert(len(ent) == len(pos))

            exampleD = exampleDs[sentence_index]
            exampleD['pos_index'] = pos_indices
            exampleD['pos'] = pos

            exampleD['dependency_index'] = dependency_indices
            exampleD['dependency'] = dependency

            exampleD['ent_index'] = ent_indices
            exampleD['ent'] = ent
            exampleD['reverse_ent'] = reverse_ent
            exampleD['ent_pos'] = ent_pos
           # else:
            '''
            ent, ent_indices, ent_words, ent_pos, reverse_ent = [], [], [], [], []
            exampleD['ent_index'] = ent_indices
            exampleD['ent'] = ent
            exampleD['reverse_ent'] = reverse_ent
            exampleD['ent_pos'] = ent_pos
            '''


            if len(verb_indices) != 0:
                exampleD['verb_index'] = verb_indices
            else:
                exampleD['verb_index'] = [0]
            exampleD['verb'] = verb

    for exampleD in exampleDs:
        example = data.Example.fromdict(exampleD, fields)
        examples.append(example)
    return examples, orig_sentences


def process_data_new(hparams, predict_sentences=None):
    train_fp, dev_fp, test_fp = hparams.train_fp, hparams.dev_fp, hparams.test_fp
    hparams.bos_token_id, hparams.eos_token_id = 101, 102
    # print("in process_data_new")

    do_lower_case = 'uncased' in hparams.model_str
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_str, do_lower_case=do_lower_case, use_fast=True,
                                              data_dir='data/pretrained_cache',
                                              add_special_tokens=False,
                                              additional_special_tokens=['[unused1]', '[unused2]', '[unused3]'])

    nlp = spacy.load("en_core_web_sm")
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    TEXT = data.Field(use_vocab=False, batch_first=True, pad_token=pad_index)
    WORD_STARTS = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    POS = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    POS_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    ENT = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    ENT_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    REVERSE_ENT = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    ENT_POS = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    VERB = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    VERB_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    DEPENDENCY = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    DEPENDENCY_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
    META_DATA = data.Field(sequential=False)
    VERB_WORDS = data.Field(sequential=False)
    POS_WORDS = data.Field(sequential=False)
    ENT_WORDS = data.Field(sequential=False)
    DEPENDENCY_WORDS = data.Field(sequential=False)
    LABELS = data.NestedField(data.Field(use_vocab=False, batch_first=True, pad_token=-100), use_vocab=False)

    fields = {'text': ('text', TEXT), 'labels': ('labels', LABELS), 'word_starts': (
        'word_starts', WORD_STARTS), 'meta_data': ('meta_data', META_DATA)}
    #if 'predict' not in hparams.mode:
    fields['pos'] = ('pos', POS)
    fields['pos_index'] = ('pos_index', POS_INDEX)
    fields['verb'] = ('verb', VERB)
    fields['verb_index'] = ('verb_index', VERB_INDEX)
    fields['ent'] = ('ent', ENT)
    fields['ent_index'] = ('ent_index', ENT_INDEX)
    fields['reverse_ent'] = ('reverse_ent', REVERSE_ENT)
    fields['ent_pos'] = ('ent_pos', ENT_POS)
    fields['dependency'] = ('dependency', DEPENDENCY)
    fields['dependency_index'] = ('dependency_index', DEPENDENCY_INDEX)

    if hparams.task == 'oie':
        label_dict = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                      'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
    else:  # hparams.task == 'conj':
        label_dict = {'CP_START': 2, 'CP': 1,
                      'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}
    #sentences_ent_tagged_train = read_tagged_dataset("sner/entities_tagged_train")
    #sentences_ent_tagged_dev = read_tagged_dataset("sner/entities_tagged_dev")
    #sentences_ent_tagged_test = read_tagged_dataset("sner/entities_tagged_test")
    # sentences_ent_tagged_predict = read_tagged_dataset("sner/PROJECTS_relation-extraction_cois_segmented_comma_replaced_ent_tagged.txt")
    #if 'predict' in hparams.mode:
        # if hparams.predict_ent!="none":
    #    sentences_ent_tagged_predict = read_tagged_dataset(hparams.predict_ent)
        # print("read senetnces ent predict: "+str(sentences_ent_tagged_predict))
        # print("read sentences_ent_tagged_train, length is: "+str(len(sentences_ent_tagged_train)))
        # print("read sentences_ent_tagged_test, length is: "+str(len(sentences_ent_tagged_test)))
        # print("read sentences_ent_tagged_dev, length is: "+str(len(sentences_ent_tagged_dev)))

    cached_train_fp, cached_dev_fp, cached_test_fp = f'{train_fp}.{hparams.model_str.replace("/", "_")}.pkl', f'{dev_fp}.{hparams.model_str.replace("/", "_")}.pkl', f'{test_fp}.{hparams.model_str.replace("/", "_")}.pkl'

    all_sentences = []
    if 'predict' in hparams.mode:
        # no caching used in predict mode
        if predict_sentences == None:  # predict
            if hparams.inp != None:
                predict_f = open(hparams.inp, 'r', encoding="utf-8")
            else:
                predict_f = open(hparams.predict_fp, 'r', encoding="utf-8")
            predict_lines = predict_f.readlines()
            fullstops = []
            predict_sentences = []

            for line in predict_lines:
                # Normalize the quotes - similar to that in training data
                line = line.replace('’', '\'')
                line = line.replace('”', '\'\'')
                line = line.replace('“', '\'\'')

                tokenized_line = line.split()
                # if len(tokenized_line) >= 512:
                    #print("length of the sentence is more than 512: " + line)
                predict_sentences.append(
                    ' '.join(tokenized_line) + ' [unused1] [unused2] [unused3]')
                predict_sentences.append('\n')
                # print(predict_sentences)

        predict_examples, all_sentences = _process_data_new(predict_sentences, hparams, fields, tokenizer, label_dict,
                                                            nlp)
        META_DATA.build_vocab(data.Dataset(predict_examples, fields=fields.values()))
       # print("printing meta_data.vocab")
        # for e in META_DATA.vocab:
        #	print(str(e)+"\t"+str(META_DATA.vocab[e]))

        predict_dataset = [(len(ex.text), idx, ex, fields) for idx, ex in enumerate(predict_examples)]
        train_dataset, dev_dataset, test_dataset = predict_dataset, predict_dataset, predict_dataset
        #print("predict set")
    else:
        if not os.path.exists(cached_train_fp) or hparams.build_cache:
            train_examples, _ = _process_data_new(train_fp, hparams, fields, tokenizer, label_dict, nlp)
            pickle.dump(train_examples, open(cached_train_fp, 'wb'))
            #print("1")
        else:
            train_examples = pickle.load(open(cached_train_fp, 'rb'))
            #print("2")
            #print(cached_train_fp)

        if not os.path.exists(cached_dev_fp) or hparams.build_cache:
            dev_examples, _ = _process_data_new(dev_fp, hparams, fields, tokenizer, label_dict, nlp)
            pickle.dump(dev_examples, open(cached_dev_fp, 'wb'))
            #print("3")
        else:
            dev_examples = pickle.load(open(cached_dev_fp, 'rb'))
            #print("4")

        if not os.path.exists(cached_test_fp) or hparams.build_cache:
            test_examples, _ = _process_data_new(test_fp, hparams, fields, tokenizer, label_dict, nlp)
            pickle.dump(test_examples, open(cached_test_fp, 'wb'))
            #print("4")
        else:
            test_examples = pickle.load(open(cached_test_fp, 'rb'))
            #print("5")
        #print(data.Dataset(train_examples, fields=fields.values()).fields)

        META_DATA.build_vocab(data.Dataset(train_examples, fields=fields.values()), data.Dataset(
            dev_examples, fields=fields.values()), data.Dataset(test_examples, fields=fields.values()))

        train_dataset = [(len(ex.text), idx, ex, fields) for idx, ex in enumerate(train_examples)]
        dev_dataset = [(len(ex.text), idx, ex, fields) for idx, ex in enumerate(dev_examples)]
        test_dataset = [(len(ex.text), idx, ex, fields) for idx, ex in enumerate(test_examples)]
        train_dataset.sort()  # to simulate bucket sort (along with pad_data)

    return train_dataset, dev_dataset, test_dataset, META_DATA.vocab, all_sentences


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def pad_data_with_ent(data):
    fields = data[0][-1]
    # print("this is data in pad_data_with_ent: "+str(data))
    TEXT = fields['text'][1]
    # print("this is ex[2]: "+str(ex[2]))
    # print("in pad_data_with_ent")
    # for ex in data:
    #	print("this is ex[3]: "+str(ex[3]))
    text_list = [ex[2].text for ex in data]
    # print("text_list: "+str(text_list))
    padded_text = torch.tensor(TEXT.pad(text_list))

    LABELS = fields['labels'][1]
    labels_list = [ex[2].labels for ex in data]
    # max_depth = max([len(l) for l in labels_list])
    max_depth = 5
    for i in range(len(labels_list)):
        pad_depth = max_depth - len(labels_list[i])
        num_words = len(labels_list[i][0])
        # print(num_words, pad_depth)
        labels_list[i] = labels_list[i] + [[0] * num_words] * pad_depth
    # print(labels_list)
    padded_labels = torch.tensor(LABELS.pad(labels_list))

    WORD_STARTS = fields['word_starts'][1]
    word_starts_list = [ex[2].word_starts for ex in data]
    padded_word_starts = torch.tensor(WORD_STARTS.pad(word_starts_list))

    META_DATA = fields['meta_data'][1]
    meta_data_list = [META_DATA.vocab.stoi[ex[2].meta_data] for ex in data]
    padded_meta_data = torch.tensor(META_DATA.pad(meta_data_list))

    paddedD = {'text': padded_text, 'labels': padded_labels,
               'word_starts': padded_word_starts, 'meta_data': padded_meta_data}

    if 'pos' in fields:
        POS = fields['pos'][1]
        pos_list = [ex[2].pos for ex in data]

        padded_pos = torch.tensor(POS.pad(pos_list))
        paddedD['pos'] = padded_pos

        POS_INDEX = fields['pos_index'][1]
        pos_index_list = [ex[2].pos_index for ex in data]
        padded_pos_index = torch.tensor(POS_INDEX.pad(pos_index_list))
        paddedD['pos_index'] = padded_pos_index

    if 'verb' in fields:
        VERB = fields['verb'][1]
        verb_list = [ex[2].verb for ex in data]
        padded_verb = torch.tensor(VERB.pad(verb_list))
        paddedD['verb'] = padded_verb

        VERB_INDEX = fields['verb_index'][1]
        verb_index_list = [ex[2].verb_index for ex in data]
        padded_verb_index = torch.tensor(VERB_INDEX.pad(verb_index_list))
        paddedD['verb_index'] = padded_verb_index

    if 'ent' in fields:
        ENT = fields['ent'][1]
        ent_list = [ex[2].ent for ex in data]
        # print("ent_list: "+str(ent_list))
        padded_ent = torch.tensor(ENT.pad(ent_list))
        paddedD['ent'] = padded_ent

        ENT_INDEX = fields['ent_index'][1]
        ent_index_list = [ex[2].ent_index for ex in data]
        padded_ent_index = torch.tensor(ENT_INDEX.pad(ent_index_list))
        paddedD['ent_index'] = padded_ent_index

        REVERSE_ENT = fields['reverse_ent'][1]
        reverse_ent_list = [ex[2].reverse_ent for ex in data]
        padded_reverse_ent = torch.tensor(REVERSE_ENT.pad(reverse_ent_list))
        paddedD['reverse_ent'] = padded_reverse_ent

        ENT_POS = fields['ent_pos'][1]
        ent_pos_list = [ex[2].ent_pos for ex in data]
        padded_ent_pos = torch.tensor(ENT_POS.pad(ent_pos_list))
        paddedD['ent_pos'] = padded_ent_pos

    if 'dependency' in fields:
        DEPENDENCY = fields['dependency'][1]
        dependency_list = [ex[2].dependency for ex in data]
        # print("ent_list: "+str(ent_list))
        padded_dependency = torch.tensor(DEPENDENCY.pad(dependency_list))
        paddedD['dependency'] = padded_dependency

        DEPENDENCY_INDEX = fields['dependency_index'][1]
        dependency_index_list = [ex[2].dependency_index for ex in data]
        padded_dependency_index = torch.tensor(DEPENDENCY_INDEX.pad(dependency_index_list))
        paddedD['dependency_index'] = padded_dependency_index
    return paddedD


def ext_to_string(extraction):
    ext_str = ''
    ext_str = f'{extraction.confidence:.02f}: ({extraction.args[0]}; {extraction.pred})'
    if len(extraction.args) >= 2:
        ext_str = f'{ext_str[:-1]}; {"; ".join(extraction.args[1:])})'
    return ext_str


def ext_to_sentence(extraction):
    ext_str = ''
    ext_str = f'{extraction.args[0]} {extraction.pred}'
    if len(extraction.args) >= 2:
        ext_str = f'{ext_str} {" ".join(extraction.args[1:])}'
    return ext_str


def coords_to_string(conj_coords, words):
    conj_str = ''
    for coord_index in conj_coords:
        if conj_coords[coord_index] == None:
            conj_str += words[coord_index] + ': None  \n'
            continue
        cc_word = words[conj_coords[coord_index].cc]
        conj_str += cc_word + ': '
        for conjunct in conj_coords[coord_index].conjuncts:
            conjunct_words = ' '.join(words[conjunct[0]:conjunct[1] + 1])
            conj_str += conjunct_words + '; '
        conj_str = conj_str[:-2] + '  \n'
    return conj_str


def convert_to_namespace(d):
    params = argparse.Namespace()
    for key in d:
        setattr(params, key, d[key])
    return params


def override_args(loaded_hparams_dict, current_hparams_dict, cline_sys_args):
    # override the values of loaded_hparams_dict with the values i current_hparams_dict
    # (only the keys in cline_sys_args)
    for arg in cline_sys_args:
        if '--' in arg:
            key = arg[2:]
            loaded_hparams_dict[key] = current_hparams_dict[key]

    for key in current_hparams_dict:
        if key not in loaded_hparams_dict:
            loaded_hparams_dict[key] = current_hparams_dict[key]

    return loaded_hparams_dict


def coords_to_sentences(conj_coords, words):
    for k in list(conj_coords):
        if conj_coords[k] is None:
            conj_coords.pop(k)

    for k in list(conj_coords):
        if words[conj_coords[k].cc] in ['nor', '&']:  # , 'or']:
            conj_coords.pop(k)

    num_coords = len(conj_coords)
    # for k in list(conj_coords):
    #     if len(conj_coords[k].conjuncts) < 3 and words[conj_coords[k].cc].lower() == 'and':
    #         conj_coords.pop(k)
    # if len(conj_coords[k].conjuncts) < 3:
    #     conj_coords.pop(k)
    # else:
    #     named_entity = False
    #     for conjunct in conj_coords[k].conjuncts:
    #         # if not words[conjunct[0]][0].isupper():
    #         if (conjunct[1]-conjunct[0]) > 0 or len(conj_coords)>1:
    #             named_entity = True
    #     if named_entity:
    #         # conj_words = []
    #         # for conjunct in conj_coords[k].conjuncts:
    #         #     conj_words.append(' '.join(words[conjunct[0]:conjunct[1]+1]))
    #         # open('temp.txt', 'a').write('\n'+' '.join(words)+'\n'+'\n'.join(conj_words)+'\n')
    #         conj_coords.pop(k)

    remove_unbreakable_conjuncts(conj_coords, words)

    conj_words = []
    for k in list(conj_coords):
        for conjunct in conj_coords[k].conjuncts:
            conj_words.append(' '.join(words[conjunct[0]:conjunct[1] + 1]))

    sentence_indices = []
    for i in range(0, len(words)):
        sentence_indices.append(i)

    roots, parent_mapping, child_mapping = get_tree(conj_coords)

    q = list(roots)

    sentences = []
    count = len(q)
    new_count = 0

    conj_same_level = []

    while (len(q) > 0):

        conj = q.pop(0)
        count -= 1
        conj_same_level.append(conj)

        for child in child_mapping[conj]:
            q.append(child)
            new_count += 1

        if count == 0:
            get_sentences(sentences, conj_same_level,
                          conj_coords, sentence_indices)
            count = new_count
            new_count = 0
            conj_same_level = []

    word_sentences = [' '.join([words[i] for i in sorted(sentence)]) for sentence in sentences]

    return word_sentences, conj_words, sentences
    # return '\n'.join(word_sentences) + '\n'


def get_tree(conj):
    parent_child_list = []

    child_mapping, parent_mapping = {}, {}

    for key in conj:
        assert conj[key].cc == key
        parent_child_list.append([])
        for k in conj:
            if conj[k] is not None:
                if is_parent(conj[key], conj[k]):
                    parent_child_list[-1].append(k)

        child_mapping[key] = parent_child_list[-1]

    parent_child_list.sort(key=list.__len__)

    for i in range(0, len(parent_child_list)):
        for child in parent_child_list[i]:
            for j in range(i + 1, len(parent_child_list)):
                if child in parent_child_list[j]:
                    parent_child_list[j].remove(child)

    for key in conj:
        for child in child_mapping[key]:
            parent_mapping[child] = key

    roots = []
    for key in conj:
        if key not in parent_mapping:
            roots.append(key)

    return roots, parent_mapping, child_mapping


def is_parent(parent, child):
    min = child.conjuncts[0][0]
    max = child.conjuncts[-1][-1]

    for conjunct in parent.conjuncts:
        if conjunct[0] <= min and conjunct[1] >= max:
            return True
    return False


def get_sentences(sentences, conj_same_level, conj_coords, sentence_indices):
    for conj in conj_same_level:

        if len(sentences) == 0:

            for conj_structure in conj_coords[conj].conjuncts:
                sentence = []
                for i in range(conj_structure[0], conj_structure[1] + 1):
                    sentence.append(i)
                sentences.append(sentence)

            min = conj_coords[conj].conjuncts[0][0]
            max = conj_coords[conj].conjuncts[-1][-1]

            for sentence in sentences:
                for i in sentence_indices:
                    if i < min or i > max:
                        sentence.append(i)

        else:
            to_add = []
            to_remove = []

            for sentence in sentences:
                if conj_coords[conj].conjuncts[0][0] in sentence:
                    sentence.sort()

                    min = conj_coords[conj].conjuncts[0][0]
                    max = conj_coords[conj].conjuncts[-1][-1]

                    for conj_structure in conj_coords[conj].conjuncts:
                        new_sentence = []
                        for i in sentence:
                            if i in range(conj_structure[0], conj_structure[1] + 1) or i < min or i > max:
                                new_sentence.append(i)

                        to_add.append(new_sentence)

                    to_remove.append(sentence)

            for sent in to_remove:
                sentences.remove(sent)
            sentences.extend(to_add)


def remove_unbreakable_conjuncts(conj, words):
    unbreakable_indices = []

    unbreakable_words = ["between", "among", "sum", "total", "addition", "amount", "value", "aggregate", "gross",
                         "mean", "median", "average", "center", "equidistant", "middle"]

    for i, word in enumerate(words):
        if word.lower() in unbreakable_words:
            unbreakable_indices.append(i)

    to_remove = []
    span_start = 0

    for key in conj:
        span_end = conj[key].conjuncts[0][0] - 1
        for i in unbreakable_indices:
            if span_start <= i <= span_end:
                to_remove.append(key)
        span_start = conj[key].conjuncts[-1][-1] + 1

    for k in set(to_remove):
        conj.pop(k)

