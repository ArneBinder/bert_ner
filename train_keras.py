import argparse
import logging

import keras
from keras import Input, Model
from keras.layers import LSTM, Dense, Bidirectional
from keras.optimizers import Adam
from keras.utils import plot_model, multi_gpu_model
import numpy as np
import sklearn.metrics as sklm
from keras_preprocessing import sequence
from tensorflow.python.client import device_lib
import tensorflow as tf

from data_load import ConllDataset, get_logger
logger = get_logger(__name__)

# taken from https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
class Metrics(keras.callbacks.Callback):
    def __init__(self, val_is_heads):
        super(Metrics, self).__init__()
        self.val_is_heads_mask = val_is_heads.reshape((-1,)).astype(bool)

    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        # flatten and mask for heads
        score = score.reshape((-1, score.shape[-1]))[self.val_is_heads_mask]
        predict = predict.reshape((-1, predict.shape[-1]))[self.val_is_heads_mask]
        targ = targ.reshape((-1, targ.shape[-1]))[self.val_is_heads_mask]

        #self.auc.append(sklm.roc_auc_score(targ, score))
        #self.confusion.append(sklm.confusion_matrix(targ, predict))
        #if sum(np.max(predict, axis=-1)) < len(predict):
        #print('WARNING: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.')
        self.precision.append(sklm.precision_score(targ, predict, average='macro'))
        #if sum(np.max(targ, axis=-1)) < len(targ):
        #print('WARNING: Recall is ill-defined and being set to 0.0 in labels with no true samples.')
        self.recall.append(sklm.recall_score(targ, predict, average='macro'))
        if self.precision[-1] == 0.0 or self.recall[-1] == 0.0:
            self.f1s.append(0.0)
        else:
            self.f1s.append(2 * self.precision[-1] * self.recall[-1] / (self.precision[-1] + self.recall[-1]))
        #self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        print('val_f1: %f — val_precision: %f — val_recall %f' % (self.f1s[-1], self.precision[-1], self.recall[-1]))
        return

def get_bi_lstm(n_hidden=768, dropout=0.0, recurrent_dropout=0.0):
    return Bidirectional(LSTM(n_hidden // 2, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

def get_model(n_classes, input_shape, input_dtype, lr, top_rnns=True):
    input = Input(shape=input_shape, dtype=input_dtype, name='bert_encodings')
    X = input
    if top_rnns:
        X = get_bi_lstm()(X)
        X = get_bi_lstm()(X)
    pred = Dense(n_classes, activation='softmax')(X)
    model_save = Model(input, pred)
    #logger.debug(f'available training devices:\n{device_lib.list_local_devices()}'.replace('\n', '\n\t'))
    devices = device_lib.list_local_devices()
    # take gpu count from device info manually, because virtual devices (e.g. XLA_GPU) cause wrong number
    gpus = len([None for d in devices if d.device_type == 'GPU'])
    if gpus > 1:
        model = multi_gpu_model(model_save, gpus=gpus, cpu_relocation=True)
        logging.info(f"Training using {gpus} GPUs...")
    else:
        model = model_save
        logging.info("Training using single GPU or CPU...")

    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer
                  )
    plot_model(model, to_file='model.png', show_shapes=True)
    # use model_save to save weights
    return model, model_save


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    #parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--explainable", dest="explainable", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    #parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    parser.add_argument("--use_default_tagset", dest="use_default_tagset", action="store_true")
    parser.add_argument("--predict_tag", dest="predict_tag", type=str, default='ner')
    args = parser.parse_args()

    # disables many logging spam
    tf.logging.set_verbosity(tf.logging.ERROR)
    # set root logging level
    logging.getLogger().setLevel(logging.DEBUG)

    # mapping from tag type to position [column] index in the dataset
    tag_types = {'ner': 3, 'pos': 1}
    assert args.predict_tag in tag_types, \
        f'the tag type to predict [{args.predict_tag}] is not in tag_types that are taken from the datasets: ' \
        f'{", ".join(tag_types.keys())}'
    logger.info('Loading data...')
    eval_dataset = ConllDataset(args.validset, tag_types=tag_types)
    train_dataset = ConllDataset(args.trainset, tag_types=tag_types)
    maxlen = max(train_dataset.maxlen, eval_dataset.maxlen)
    train_dataset.seqlen = maxlen
    eval_dataset.seqlen = maxlen
    #tagset = eval_dataset.tagset
    logger.info('encode tokens with BERT...')
    x_train_encoded = train_dataset.x_bertencoded()
    x_eval_encoded = eval_dataset.x_bertencoded()
    assert x_train_encoded.shape[1:] == x_eval_encoded.shape[1:], 'shape mismatch for bert encoded sequences'
    bert_output_shape = x_train_encoded.shape[1:]
    bert_output_dtype = x_train_encoded.dtype

    default_tagsets = {'ner': ('O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG'),
                       }
    tagset = eval_dataset.generate_y_and_tagset(tag_type=args.predict_tag, tagset=('<PAD>',) + default_tagsets[args.predict_tag] if args.use_default_tagset else None)
    _ = train_dataset.generate_y_and_tagset(tag_type=args.predict_tag, tagset=tagset)

    logger.info('Build model...')
    model, get_model = get_model(n_classes=len(tagset), input_shape=bert_output_shape, input_dtype=bert_output_dtype,
                                 lr=args.lr, top_rnns=args.top_rnns)

    logger.info('Train with batch_size=%i...' % args.batch_size)
    metrics = Metrics(val_is_heads=sequence.pad_sequences(eval_dataset.is_heads, maxlen=maxlen))
    model.fit(x_train_encoded, train_dataset.y,
              batch_size=args.batch_size,
              epochs=args.n_epochs,
              validation_data=(x_eval_encoded, eval_dataset.y),
              callbacks=[metrics],
              #verbose=0
              )
    #score, acc, f1 = model.evaluate(x_eval_encoded, y_eval,
    #                           batch_size=args.batch_size)
    #print('Test score:', score)
    #print('Test acc:', acc)
    #print('Test f1:', f1)
    #eval_pred = model.predict(x_eval_encoded)

    #logger.info('done')