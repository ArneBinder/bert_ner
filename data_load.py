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
from itertools import chain

import numpy as np
import torch
from torch.utils import data
from keras_preprocessing import sequence

from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


class ConllDataset(data.Dataset):
    def __init__(self, fpath, bert_model=None, tagset=None, cache_dir='cache'):
        """
        fpath: [train|valid|test].txt
        """
        self.bert_model = bert_model

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print('process data from: %s...' % fpath)
        self.cache_base_fn = os.path.join(self.cache_dir, f'{fpath.split("/")[-1]}_')

        all_data = self.calc_cached(func=self.load_dataset, fn='dataset.json', fpath=fpath, tagset=tagset)
        for k, v in all_data.items():
            self.__setattr__(k, v)

    def load_dataset(self, fpath, tagset):
        entries = open(fpath, 'r').read().strip().split("\n\n")
        maxlen = -1
        words_str = []
        x = []
        is_heads = []
        tags_str = []
        t = []
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sent = ["[CLS]"] + words + ["[SEP]"]
            sent_tags = ["<PAD>"] + tags + ["<PAD>"]
            c_words, c_x, c_is_heads, c_tags, c_t = self.convert_to_record(sent, sent_tags)
            words_str.append(c_words)
            x.append(c_x)
            is_heads.append(c_is_heads)
            tags_str.append(c_tags)
            t.append(c_t)
            maxlen = max(len(c_x), maxlen)
        print('loaded %i entries. maxlen: %i' % (len(x), maxlen))

        # create vocab from all tags (move padding tag to idx=0)
        if tagset is None:
            tagset = ['<PAD>'] + list(set(chain(*t)) - {'<PAD>'})
        tag2idx = {tag: idx for idx, tag in enumerate(tagset)}
        #idx2tag = {idx: tag for idx, tag in enumerate(tagset)}
        print('convert tags to indices...')
        y = []
        for i, tags in enumerate(t):
            assert len(x[i]) == len(
                tags), f'number of tags [{len(self.x[i])}] does not match number of tokens {len(tags)}'
            yy = [tag2idx[t] for t in tags]
            y.append(yy)

        return {'words_str': words_str,
                    'x': x,
                    'is_heads': is_heads,
                    'tags_str': tags_str,
                    'y': y,
                    'tagset': tagset,
                    'maxlen': maxlen}


    def calc_cached(self, func, fn, *args, **kwargs):
        full_fn = self.cache_base_fn + fn
        if fn.endswith('.json'):
            if os.path.exists(full_fn):
                print(f'load from cache {full_fn}...')
                res = json.load(open(full_fn))
            else:
                res = func(*args, **kwargs)
                print(f'save to cache {full_fn}...')
                json.dump(res, open(full_fn, 'w'))
        elif fn.endswith('.pkl'):
            if os.path.exists(full_fn):
                print(f'load from cache {full_fn}...')
                res = pickle.load(open(full_fn))
            else:
                res = func(*args, **kwargs)
                print(f'save to cache {full_fn}...')
                pickle.dump(res, open(full_fn, 'wb'))
        elif fn.endswith('.npy'):
            if os.path.exists(full_fn):
                print(f'load from cache {full_fn}...')
                res = np.load(full_fn)
            else:
                res = func(*args, **kwargs)
                print(f'save to cache {full_fn}...')
                np.save(res, full_fn)
        else:
            raise NotImplementedError(f'Unknown cache file extension: {fn}. Use either json, pkl or npy.')
        return res


    def convert_to_record(self, words, tags):
        # We give credits only to the first piece.
        x, t = [], []  # list of ids
        is_heads = []  # list. 1: the token is the first piece of a word
        for word, tag in zip(words, tags):
            tokens = tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)

            tag = [tag] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            #yy = [self.tag2idx[each] for each in tag]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            t.extend(tag)

        assert len(x) == len(t) == len(is_heads), f"len(x)={len(x)}, len(t)={len(t)}, len(is_heads)={len(is_heads)}"

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.words_str[idx], self.x[idx], self.is_heads[idx], self.tags_str[idx], self.y[idx], self.seqlen[idx]

    def pad_to_numpy(self, maxlen: int, padding='post'):
        print('pad x, y and is_heads...')
        self.x = sequence.pad_sequences(self.x, maxlen=maxlen, padding=padding)
        self.y = sequence.pad_sequences(self.y, maxlen=maxlen, padding=padding)
        self.is_heads = sequence.pad_sequences(self.is_heads, maxlen=maxlen, padding=padding)

    def encode_with_bert(self, sequences: np.ndarray, return_layers=-1, batch_size=32):
        #assert self.bert_model is not None, 'no BERT model loaded'
        if self.bert_model is None:
            print('load BERT model...')
            self.bert_model = BertModel.from_pretrained('bert-base-cased').eval()
        assert isinstance(sequences, np.ndarray), 'sequences has to be an ndarray. Did you call pad_to_numpy(maxlen)?'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert_model.to(device)
        encs_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                sequences_tensor = torch.LongTensor(sequences[i:(i + batch_size)]).to(device)
                encoded_layers, _ = self.bert_model(sequences_tensor)
                encs = encoded_layers[return_layers]
                encs_list.append(encs)
        return np.concatenate(encs_list, axis=0)

    def x_bertencoded(self):
        return self.calc_cached(func=self.encode_with_bert, fn='xencoded.npy', sequences=self.x)

    def vocab_size(self):
        return len(tokenizer.vocab)



