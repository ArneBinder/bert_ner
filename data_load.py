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

#tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
#idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class ConllDataset(data.Dataset):
    def __init__(self, fpath, bert_model=None, tagset=None, cache_dir='cache'):
        """
        fpath: [train|valid|test].txt
        """
        self.bert_model = bert_model
        if self.bert_model is None:
            print('load BERT model...')
            self.bert_model = BertModel.from_pretrained('bert-base-cased').eval()

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print('process data from: %s...' % fpath)
        self.cache_base_fn = os.path.join(self.cache_dir, f'{fpath.split("/")[-1]}')
        cache_dataset_fn = self.cache_base_fn + '_dataset.json'
        if os.path.exists(cache_dataset_fn):
            print('cached dataset found. load {cache_fn}...'  )
            #loaded = pickle.load(open(cache_dataset_fn, 'rb'))
            loaded = json.load(open(cache_dataset_fn))
            for k, v in loaded.items():
                self.__setattr__(k, v)
            print(f'loaded data for keys: {", ".join(loaded.keys())}')
            self.tag2idx = {tag: idx for idx, tag in enumerate(self.tagset)}
            self.idx2tag = {idx: tag for idx, tag in enumerate(self.tagset)}
        else:
            entries = open(fpath, 'r').read().strip().split("\n\n")
            self.maxlen = -1
            self.words_str = []
            self.x = []
            self.is_heads = []
            self.tags_str = []
            self.t = []
            for entry in entries:
                words = [line.split()[0] for line in entry.splitlines()]
                tags = ([line.split()[-1] for line in entry.splitlines()])
                sent = ["[CLS]"] + words + ["[SEP]"]
                sent_tags = ["<PAD>"] + tags + ["<PAD>"]
                words, x, is_heads, tags, t = self.convert_to_record(sent, sent_tags)
                self.words_str.append(words)
                self.x.append(x)
                self.is_heads.append(is_heads)
                self.tags_str.append(tags)
                self.t.append(t)
                self.maxlen = max(len(x), self.maxlen)
            print('loaded %i entries. maxlen: %i' % (len(self.x), self.maxlen))

            self.tagset = tagset
            # create vocab from all tags (move padding tag to idx=0)
            if self.tagset is None:
                self.tagset = ['<PAD>'] + list(set(chain(*self.t)) - {'<PAD>'})
            self.tag2idx = {tag: idx for idx, tag in enumerate(self.tagset)}
            self.idx2tag = {idx: tag for idx, tag in enumerate(self.tagset)}
            print('convert tags to indices...')
            self.y = []
            for i, tags in enumerate(self.t):
                assert len(self.x[i]) == len(tags), f'number of tags [{len(self.x[i])}] does not match number of tokens {len(tags)}'
                yy = [self.tag2idx[t] for t in tags]
                self.y.append(yy)

            print(f'dump dataset to {cache_dataset_fn}...')
            all_data = {'words_str': self.words_str,
                        'x': self.x,
                        'is_heads': self.is_heads,
                        'tags_str': self.tags_str,
                        'y': self.y,
                        'tagset': self.tagset,
                        'maxlen': self.maxlen}
            json.dump(all_data, open(cache_dataset_fn, 'w'))
            #pickle.dump(all_data, open(cache_dataset_fn, 'wb'))

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
        assert self.bert_model is not None, 'no BERT model loaded'
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
        cache_fn = self.cache_base_fn + '_xencoded.npy'
        if os.path.exists(cache_fn):
            print(f'load encoded x from file: {cache_fn}')
            res = np.load(cache_fn)
        else:
            res = self.encode_with_bert(self.x)
            print(f'save encoded x to file: {cache_fn}')
            np.save(cache_fn, res)
        return res

    def vocab_size(self):
        return len(tokenizer.vocab)



