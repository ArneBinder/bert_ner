import logging
import torch
from collections import defaultdict

import keras
from keras import Input, Model
from keras.layers import LSTM, Dense, Bidirectional, K
from keras.optimizers import Adam
from keras.utils import plot_model, multi_gpu_model
from tensorflow.python.client import device_lib
import tensorflow as tf
from torch import nn, optim

from data_load import get_logger, ConllDataset

logger = get_logger(__name__)
# disables many logging spam
tf.logging.set_verbosity(tf.logging.ERROR)
# set root logging level
logging.getLogger().setLevel(logging.DEBUG)


class ConllModel(object):

    def fit(self, train_dataset: ConllDataset, batch_size: int, n_epochs: int, eval_dataset: ConllDataset):
        raise NotImplementedError('implement this method')

    def evaluate(self, test_dataset: ConllDataset, batch_size:int):
        raise NotImplementedError('implement this method')

### Keras

class ANDCounter(keras.layers.Layer):
    """
    inspired by https://github.com/keras-team/keras/issues/10884#issuecomment-412120393
    and https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras

    conditions_and is a function that maps a tuple (y_true, y_pred) to a list of conditions that are then reduced
    via logical AND along the last axis. True elements are counted and finally returned.
    """
    def __init__(self, conditions_and, name="and_counter", **kwargs):
        super(ANDCounter, self).__init__(name=name, **kwargs)
        self.stateful = True
        self.count = keras.backend.variable(value=0, dtype="int32")
        self.cond = conditions_and

    def reset_states(self):
        keras.backend.set_value(self.count, 0)

    def __call__(self, y_true, y_pred):
        # initial shape is (batch_size, squence_length, n_classes)

        conds_list = self.cond(y_true, y_pred) #+ (y_true_wo_index_mask, )

        conds_xd = K.cast(K.stack(conds_list, axis=-1), 'bool')

        res = K.sum(K.cast(K.all(conds_xd, axis=-1), 'int32'))

        updates = [
            keras.backend.update_add(
                self.count,
                res)]
        self.add_update(updates)
        return self.count

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        res = defaultdict(dict)
        for m_name in logs.keys():
            if m_name.startswith('val_'):
                res['val'][m_name[len('val_'):]] = logs[m_name]
            else:
                res['train'][m_name] = logs[m_name]
        for k in res.keys():
            current_metrics = counts_to_metrics(**res[k])
            logger.info(format_metrics(current_metrics, prefix=k))
        return

def get_bi_lstm(n_hidden=768, dropout=0.0, recurrent_dropout=0.0):
    return Bidirectional(LSTM(n_hidden // 2, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

class KerasModel(ConllModel):
    def __init__(self, n_classes, input_dims, lr, top_rnns=True, metrics_eval_discard_first_classes=2):
        self.train_history = None
        input = Input(shape=(None, input_dims), dtype='float32', name='bert_encodings')
        X = input
        if top_rnns:
            X = get_bi_lstm()(X)
            X = get_bi_lstm()(X)
        pred = Dense(n_classes, activation='softmax')(X)
        self.model_save = Model(input, pred)
        #logger.debug(f'available training devices:\n{device_lib.list_local_devices()}'.replace('\n', '\n\t'))
        devices = device_lib.list_local_devices()
        # take gpu count from device info manually, because virtual devices (e.g. XLA_GPU) cause wrong number
        gpus = len([None for d in devices if d.device_type == 'GPU'])
        if gpus > 1:
            self.model = multi_gpu_model(self.model_save, gpus=gpus, cpu_relocation=True)
            logging.info(f"Training using {gpus} GPUs...")
        else:
            self.model = self.model_save
            logging.info("Training using single GPU or CPU...")

        optimizer = Adam(lr=lr)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[ANDCounter(conditions_and=lambda y_true, y_pred: (y_true,
                                                                                 K.round(y_pred),
                                                                                 # This condition masks all entries where y_true has class=0, i.e. <PAD>:
                                                                                 #   1) gold values, except for the first class, are summed along the class-axis
                                                                                 #   2) the resulting vector is broadcast back to the original format (via stack and number of classes)
                                                                                 K.stack([K.sum(y_true[:, :, metrics_eval_discard_first_classes:],
                                                                                          axis=-1)] * n_classes, axis=-1),
                                                                                 ),
                                          name='tp'),
                               ANDCounter(conditions_and=lambda y_true, y_pred: (K.abs(y_true - K.ones_like(y_true)),
                                                                                 K.round(y_pred),
                                                                                 # this condition masks all entries where y_true has class=0, i.e. <PAD> (see above)
                                                                                 K.stack([K.sum(y_true[:, :, metrics_eval_discard_first_classes:],
                                                                                          axis=-1)] * n_classes, axis=-1),
                                                                                 ),
                                          name='fp'),
                               ANDCounter(conditions_and=lambda y_true, y_pred: (y_true,
                                                                                 K.abs(K.round(y_pred) - K.ones_like(y_pred)),
                                                                                 # this condition masks all entries where y_true has class=0, i.e. <PAD> (see above)
                                                                                 K.stack([K.sum(y_true[:, :, metrics_eval_discard_first_classes:],
                                                                                          axis=-1)] * n_classes, axis=-1),
                                                                                 ),
                                          name='fn'),
                               ANDCounter(conditions_and=lambda y_true, y_pred: (y_true,
                                                                                 # this condition masks all entries where y_true has class=0, i.e. <PAD> (see above)
                                                                                 K.stack([K.sum(y_true[:, :, metrics_eval_discard_first_classes:],
                                                                                          axis=-1)] * n_classes, axis=-1),
                                                                                 ),
                                          name='total_count'),
                               'acc', ]
                      )
        plot_model(self.model, to_file='model.png', show_shapes=True)


    def fit(self, train_dataset, batch_size, n_epochs, eval_dataset):
        train_history = self.model.fit(x=train_dataset.x_bertencoded(), y=train_dataset.y,
                                  batch_size=batch_size,
                                  epochs=n_epochs,
                                  validation_data=(eval_dataset.x_bertencoded(), eval_dataset.y),
                                  callbacks=[Metrics()]
                                  # verbose=0
                                  )
        return get_metrics_from_hist(train_history.history)

    def evaluate(self, test_dataset, batch_size):
        test_metrics_list = self.model.evaluate(test_dataset.x_bertencoded(), test_dataset.y, batch_size=batch_size)
        test_metrics = {self.model.metrics_names[i]: m for i, m in enumerate(test_metrics_list)}
        return test_metrics

def get_metrics_from_hist(history, idx=-1):
    res = defaultdict(dict)
    for m_name in history.keys():
        if m_name.startswith('val_'):
            res['val'][m_name[len('val_'):]] = history[m_name][idx]
        else:
            res['train'][m_name] = history[m_name][idx]
    return res

### Pytorch

class PytorchNet(nn.Module):
    def __init__(self, n_classes, input_dims, top_rnns=False, device='cpu'):
        super().__init__()

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=input_dims, hidden_size=input_dims // 2, batch_first=True)
        self.fc = nn.Linear(input_dims, n_classes)

        self.device = device


    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.top_rnns:
            x, _ = self.rnn(x)
        logits = self.fc(x)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

class PytorchModel(ConllModel):
    def __init__(self, n_classes, input_dims, lr, top_rnns=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PytorchNet(n_classes=n_classes, input_dims=input_dims, device=device, top_rnns=top_rnns).to(device)
        model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # TODO


### general purpose

def counts_to_metrics(tp, fp, fn, **unused):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {'f1': round(2 * ((precision * recall) / (precision + recall)), 3), 'precision': round(precision, 3), 'recall': round(recall, 3)}

def format_metrics(metrics, prefix):
    return ' — '.join([f'{prefix}_{m_name}: {metrics[m_name]}' for m_name in sorted(metrics.keys())])