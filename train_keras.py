import argparse
import logging
from collections import defaultdict

import keras
from keras import Input, Model
from keras.layers import LSTM, Dense, Bidirectional, K
from keras.optimizers import Adam
from keras.utils import plot_model, multi_gpu_model
from tensorflow.python.client import device_lib
import tensorflow as tf

from data_load import ConllDataset, get_logger
logger = get_logger(__name__)

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

def get_model(n_classes, input_shape, input_dtype, lr, top_rnns=True, metrics_eval_discard_first_classes=2):
    input = Input(shape=(None, input_shape[-1]), dtype=input_dtype, name='bert_encodings')
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
    plot_model(model, to_file='model.png', show_shapes=True)
    # use model_save to save weights
    return model, model_save

def counts_to_metrics(tp, fp, fn, **unused):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {'f1': round(2 * ((precision * recall) / (precision + recall)), 3), 'precision': round(precision, 3), 'recall': round(recall, 3)}

def get_metrics_from_hist(history, idx=-1):
    res = defaultdict(dict)
    for m_name in history.keys():
        if m_name.startswith('val_'):
            res['val'][m_name[len('val_'):]] = history[m_name][idx]
        else:
            res['train'][m_name] = history[m_name][idx]
    return res

def format_metrics(metrics, prefix):
    return ' — '.join([f'{prefix}_{m_name}: {metrics[m_name]}' for m_name in sorted(metrics.keys())])

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
    parser.add_argument("--testset", type=str, default="conll2003/test.txt")
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
    train_dataset.maxlen = maxlen
    eval_dataset.maxlen = maxlen
    #tagset = eval_dataset.tagset
    logger.info('encode tokens with BERT...')
    x_train_encoded = train_dataset.x_bertencoded()
    x_eval_encoded = eval_dataset.x_bertencoded()
    assert x_train_encoded.shape[1:] == x_eval_encoded.shape[1:], 'shape mismatch for bert encoded sequences'
    bert_output_shape = x_train_encoded.shape[1:]
    bert_output_dtype = x_train_encoded.dtype

    default_tagsets = {'ner': ('O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG'),
                       }
    tagset = eval_dataset.generate_y_and_tagset(tag_type=args.predict_tag,
                                                tagset=('<PAD>',) + default_tagsets[args.predict_tag] if args.use_default_tagset else None,
                                                to_categorical=True)
    _ = train_dataset.generate_y_and_tagset(tag_type=args.predict_tag, tagset=tagset,
                                            to_categorical=True)

    # TODO: put original pytorch model on top to recreate original performance
    logger.info('Build model...')
    model, save_model = get_model(n_classes=len(tagset), input_shape=bert_output_shape, input_dtype=bert_output_dtype,
                                  lr=args.lr, top_rnns=args.top_rnns)

    logger.info('Train with batch_size=%i...' % args.batch_size)
    train_history = model.fit(x=x_train_encoded, y=train_dataset.y,
                      batch_size=args.batch_size,
                      epochs=args.n_epochs,
                      validation_data=(x_eval_encoded, eval_dataset.y),
                      callbacks=[Metrics()]
                      #verbose=0
                      )

    train_metrics = get_metrics_from_hist(train_history.history)
    for _data, _metrics in train_metrics.items():
        logger.info(format_metrics(metrics=_metrics, prefix=_data))
        final_metrics = counts_to_metrics(**_metrics)
        logger.info(format_metrics(metrics=final_metrics, prefix=_data))

    if args.testset != '':
        logger.info('Test...')
        test_dataset = ConllDataset(args.testset, tag_types=tag_types)
        x_test_encoded = test_dataset.x_bertencoded()
        test_dataset.generate_y_and_tagset(tag_type=args.predict_tag, tagset=tagset, to_categorical=True)
        test_metrics_list = model.evaluate(x_test_encoded, test_dataset.y, batch_size=args.batch_size)
        test_metrics = {model.metrics_names[i]: m for i, m in enumerate(test_metrics_list)}
        logger.info(format_metrics(metrics=test_metrics, prefix='test'))
        test_metrics_final = counts_to_metrics(**test_metrics)
        logger.info(format_metrics(metrics=test_metrics_final, prefix='test'))

    #logger.info('done')