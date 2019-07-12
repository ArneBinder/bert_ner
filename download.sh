#!/bin/bash

train=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/a8bc4813b49cafe436cf71b49ef97c5ddae3c78c/neuroner/data/conll2003/en/train.txt
valid=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/a8bc4813b49cafe436cf71b49ef97c5ddae3c78c/neuroner/data/conll2003/en/valid.txt
test=https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/a8bc4813b49cafe436cf71b49ef97c5ddae3c78c/neuroner/data/conll2003/en/test.txt

mkdir conll2003 | wget --show-progress $train && mv train.txt conll2003
wget --show-progress $valid && mv valid.txt conll2003
wget --show-progress $test && mv test.txt conll2003