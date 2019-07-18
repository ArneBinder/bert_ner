'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import json
import os
import pickle
from collections import defaultdict
from itertools import chain
import logging

import numpy as np
import torch
from torch.utils import data
import keras
import keras_preprocessing

from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

PADDING = 'post'

def get_logger(name):
    # logging.basicConfig(level=logging.DEBUG)
    LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(LOGGING_FORMAT)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.propagate = False
    # for tensorflow logging, see: https://stackoverflow.com/questions/44853059/tensorflow-logging-messages-do-not-appear
    return logger

logger = get_logger(__name__)

class ConllDataset(data.Dataset):
    bert_model = None
    tokenizer = None

    def __init__(self, fpath, tag_types, word_idx=0, cache_dir='cache'):
        """
        fpath: [train|valid|test].txt
        """

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info('process data from: %s...' % fpath)
        self.cache_base_fn = os.path.join(self.cache_dir, f'{fpath.split("/")[-1]}_')

        all_data = self.calc_cached(func=self.load_dataset, fn='dataset.json', fpath=fpath, tag_types=tag_types,
                                    word_idx=word_idx)
        for k, v in all_data.items():
            self.__setattr__(k, v)

    def load_dataset(self, fpath, tag_types, word_idx=0):
        if ConllDataset.tokenizer is None:
            logger.info('load BERT tokenizer...')
            ConllDataset.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        entries = open(fpath, 'r').read().strip().split("\n\n")
        maxlen = -1
        words_str = []
        x = []
        is_heads = []
        #tags_str = []
        tag_padding = [{name: "<PAD>" for name in tag_types.keys()}]
        t = defaultdict(list)
        for entry in entries:
            # comment lines start with "#"
            words = [line.split()[word_idx] for line in entry.splitlines() if not line.startswith('#')]
            tags = [{name: line.split()[idx] for name, idx in tag_types.items()} for line in entry.splitlines()]
            sent_tags = tag_padding + tags + tag_padding
            sent = ["[CLS]"] + words + ["[SEP]"]
            c_words, c_x, c_is_heads, c_t = self.convert_to_record(sent, sent_tags)
            words_str.append(c_words)
            x.append(c_x)
            is_heads.append(c_is_heads)
            #tags_str.append(c_tags)
            for _name, _tags in c_t.items():
                t[_name].append(_tags)
            maxlen = max(len(c_x), maxlen)
        logger.info('loaded %i entries. maxlen: %i' % (len(x), maxlen))

        return {'words_str': words_str,
                'x': x,
                'is_heads': is_heads,
                't': t,
                #'tags_str': tags_str,
                #'tagset': tagset,
                'maxlen': maxlen}


    def calc_cached(self, func, fn, *args, **kwargs):
        full_fn = self.cache_base_fn + fn
        if fn.endswith('.json'):
            if os.path.exists(full_fn):
                logger.info(f'load from cache {full_fn}...')
                res = json.load(open(full_fn))
            else:
                res = func(*args, **kwargs)
                logger.info(f'save to cache {full_fn}...')
                json.dump(res, open(full_fn, 'w'))
        elif fn.endswith('.pkl'):
            if os.path.exists(full_fn):
                logger.info(f'load from cache {full_fn}...')
                res = pickle.load(open(full_fn))
            else:
                res = func(*args, **kwargs)
                logger.info(f'save to cache {full_fn}...')
                pickle.dump(res, open(full_fn, 'wb'))
        elif fn.endswith('.npy'):
            if os.path.exists(full_fn):
                logger.info(f'load from cache {full_fn}...')
                res = np.load(full_fn)
            else:
                res = func(*args, **kwargs)
                logger.info(f'save to cache {full_fn}...')
                np.save(full_fn, res)
        else:
            raise NotImplementedError(f'Unknown cache file extension: {fn}. Use either json, pkl or npy.')
        return res


    def convert_to_record(self, words, tags):
        # We give credits only to the first piece.
        x = []  # list of ids
        t = defaultdict(list)
        is_heads = []  # list. 1: the token is the first piece of a word
        for word, tag_dict in zip(words, tags):
            tokens = ConllDataset.tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            xx = ConllDataset.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            for _name, _tag in tag_dict.items():
                t[_name].extend([_tag] + ["<PAD>"] * (len(tokens) - 1))  # <PAD>: no decision
            #yy = [self.tag2idx[each] for each in tag]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            #t.extend(tag_dict)

        # ensure length
        assert len(x) == len(is_heads), f"len(x)={len(x)} != len(is_heads)={len(is_heads)}"
        for _name, _tags in t.items():
            assert len(x) == len(_tags), f"len(x)={len(x)} != len(tags)={len(_tags)}"

        # to string
        words = " ".join(words)
        #tags = " ".join(tags)
        return words, x, is_heads, t

    def __len__(self):
        return len(self.x)


    def generate_y_and_tagset(self, tag_type='ner', tagset=None, pad_to_numpy=True, to_categorical=True):
        # create vocab from all tags (move padding tag to idx=0)
        if tagset is None:
            tagset = ['<PAD>'] + list(set(chain(*self.t[tag_type])) - {'<PAD>'})
        tag2idx = {tag: idx for idx, tag in enumerate(tagset)}
        # idx2tag = {idx: tag for idx, tag in enumerate(tagset)}
        logger.info('convert tags to indices...')
        y = []
        for i, tags in enumerate(self.t[tag_type]):
            # is already checked during construction, see convert_to_record
            #assert len(self.x[i]) == len(tags), \
            #    f'number of tags [{len(self.x[i])}] does not match number of tokens {len(tags)}'
            yy = [tag2idx[t] for t in tags]
            y.append(yy)
        if pad_to_numpy:
            y = keras_preprocessing.sequence.pad_sequences(y, maxlen=self.seqlen, padding=PADDING)
            if to_categorical:
                y = keras.utils.to_categorical(y, num_classes=len(tagset))
        elif to_categorical:
            logger.warning('When pad_to_numpy=False, the value of to_categorical is disregarded. Can only convert a padded numpy array to categorial.')
        self.y = y
        return tagset


    def encode_with_bert(self, sequences: np.ndarray, return_layers=-1, batch_size=32):
        #assert self.bert_model is not None, 'no BERT model loaded'
        if ConllDataset.bert_model is None:
            logger.info('load BERT model...')
            ConllDataset.bert_model = BertModel.from_pretrained('bert-base-cased').eval()
        #assert isinstance(sequences, np.ndarray), 'sequences has to be an ndarray. Did you call pad_to_numpy(maxlen)?'
        if not isinstance(sequences, np.ndarray):
            sequences = keras_preprocessing.sequence.pad_sequences(sequences, maxlen=self.seqlen, padding=PADDING)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ConllDataset.bert_model.to(device)
        encs_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                sequences_tensor = torch.LongTensor(sequences[i:(i + batch_size)]).to(device)
                encoded_layers, _ = ConllDataset.bert_model(sequences_tensor)
                encs = encoded_layers[return_layers]
                encs_list.append(encs.detach().cpu().numpy())
        return np.concatenate(encs_list, axis=0)

    def x_bertencoded(self):
        return self.calc_cached(func=self.encode_with_bert, fn='xencoded.npy', sequences=self.x)

    def vocab_size(self):
        return len(ConllDataset.tokenizer.vocab)



