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
import pickle

import numpy as np
import torch
from torch.utils import data

from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

def convert_to_record(words, tags):
    # We give credits only to the first piece.
    x, y = [], []  # list of ids
    is_heads = []  # list. 1: the token is the first piece of a word
    for w, t in zip(words, tags):
        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        xx = tokenizer.convert_tokens_to_ids(tokens)

        is_head = [1] + [0] * (len(tokens) - 1)

        t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
        yy = [tag2idx[each] for each in t]  # (T,)

        x.extend(xx)
        is_heads.extend(is_head)
        y.extend(yy)

    assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

    # seqlen
    seqlen = len(y)

    # to string
    words = " ".join(words)
    tags = " ".join(tags)
    return words, x, is_heads, tags, y, seqlen

class NerDataset(data.Dataset):
    def __init__(self, fpath):
        """
        fpath: [train|valid|test].txt
        """
        self.maxlen = -1
        entries = open(fpath, 'r').read().strip().split("\n\n")
        #sents, tags_li = [], [] # list of lists
        #records = []
        self.words = []
        self.x = []
        self.is_heads = []
        self.tags = []
        self.y = []
        self.seqlen = []
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sent = ["[CLS]"] + words + ["[SEP]"]
            sent_tags = ["<PAD>"] + tags + ["<PAD>"]
            words, x, is_heads, tags, y, seqlen = convert_to_record(sent, sent_tags)
            self.words.append(words)
            self.x.append(x)
            self.is_heads.append(is_heads)
            self.tags.append(tags)
            self.y.append(y)
            self.seqlen.append(seqlen)
            self.maxlen = max(seqlen, self.maxlen)

            #records.append((words, x, is_heads, tags, y, seqlen))
            #sents.append(sent)
            #tags_li.append(sent_tags)
        #self.sents, self.tags_li = sents, tags_li
        #self.records = records


        pickle.dump([self.words, self.x, self.is_heads, self.tags, self.y, self.seqlen], open(f'dataset_{fpath.split("/")[-1]}.pkl', 'wb'))


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.words[idx], self.x[idx], self.is_heads[idx], self.tags[idx], self.y[idx], self.seqlen[idx]

    #def get_xy(self):
        #words, x, is_heads, tags, y, seqlen = zip(*self.records)
        #return list(x), list(y)

    def vocab_size(self):
        return len(tokenizer.vocab)

    def old_getitem(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch, maxlen=None):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)

    if maxlen is None:
        maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    return words, torch.LongTensor(x), is_heads, tags, torch.LongTensor(y), seqlens


