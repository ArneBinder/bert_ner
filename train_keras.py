import argparse

import keras
import torch
from keras import Input, Model
from keras.layers import LSTM, Dense, Bidirectional
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical
from keras_preprocessing import sequence
from pytorch_pretrained_bert import BertModel
import keras.backend as K
import numpy as np
import sklearn.metrics as sklm

from data_load import NerDataset, VOCAB


# taken from https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        # TODO: exclude padded items
        # TODO: how to handle <PAD> predictions?
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        score = score.reshape((-1, score.shape[-1]))
        predict = predict.reshape((-1, predict.shape[-1]))
        targ = targ.reshape((-1, targ.shape[-1]))

        #self.auc.append(sklm.roc_auc_score(targ, score))
        #self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict, average='macro'))
        self.recall.append(sklm.recall_score(targ, predict, average='macro'))
        self.f1s.append(sklm.f1_score(targ, predict, average='macro'))
        #self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        print('val_f1: %f — val_precision: %f — val_recall %f' % (self.f1s[-1], self.precision[-1], self.recall[-1]))
        return

# taken from https://datascience.stackexchange.com/questions/13746/how-to-define-a-custom-performance-metric-in-keras
# RETURNS NANs
def f1_score(y_true, y_pred):
    """
    f1 score

    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))


def encode_with_bert(bert_model, sequences, return_layers=-1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model.to(device)
    with torch.no_grad():
        sequences_tensor = torch.LongTensor(sequences).to(device)
        encoded_layers, _ = bert_model(sequences_tensor)
        encs = encoded_layers[return_layers]
    return encs.detach().cpu().numpy()


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
    args = parser.parse_args()

    print('Loading data...')
    train_dataset = NerDataset(args.trainset)
    eval_dataset = NerDataset(args.validset)
    print('train_dataset.maxlen: %i' % train_dataset.maxlen)
    print('eval_dataset.maxlen: %i' % eval_dataset.maxlen)
    maxlen = max(train_dataset.maxlen, eval_dataset.maxlen)
    assert train_dataset.vocab_size() == eval_dataset.vocab_size(), 'train_dataset.vocab_size [%i] != eval_dataset.vocab_size [%i]' % (train_dataset.vocab_size(), eval_dataset.vocab_size())
    vocab_size = train_dataset.vocab_size()
    print('tokenizer vocab size: %i' % vocab_size)

    x_train, y_train = train_dataset.get_xy()
    x_eval, y_eval = eval_dataset.get_xy()
    xy_traineval = [x_train, y_train, x_eval, y_eval]
    for i, data in enumerate(xy_traineval):
        xy_traineval[i] = sequence.pad_sequences(data, maxlen=maxlen, padding='post')
        print('data shape:', xy_traineval[i].shape)
    x_train, y_train, x_eval, y_eval = xy_traineval
    y_train = to_categorical(y_train, num_classes=len(VOCAB))
    y_eval = to_categorical(y_eval, num_classes=len(VOCAB))

    print('Embed sequences with BERT...')
    bert = BertModel.from_pretrained('bert-base-cased').eval()
    x_train_encoded = encode_with_bert(bert, x_train)
    x_eval_encoded = encode_with_bert(bert, x_eval)
    assert x_train_encoded.shape[1:] == x_eval_encoded.shape[1:], 'shape mismatch for bert encoded sequences'
    bert_output_shape = x_train_encoded.shape[1:]
    bert_output_dtype = x_train_encoded.dtype

    print('Build model...')
    input_words = Input(shape=(maxlen, ), dtype='int32', name='word_ids')
    input = Input(shape=bert_output_shape, dtype=bert_output_dtype, name='bert_encodings')
    X = input
    if args.top_rnns:
        X = Bidirectional(LSTM(768 // 2, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(X)
        X = Bidirectional(LSTM(768 // 2, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(X)
    pred = Dense(len(VOCAB), activation='softmax')(X)
    model = Model(input, pred)

    optimizer = Adam(lr=args.lr)
    #metrics = Metrics()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  #metrics=['accuracy']
                  )
    plot_model(model, to_file='model.png', show_shapes=True)


    print('Train with batch_size=%i...' % args.batch_size)
    metrics = Metrics()
    model.fit(x_train_encoded, y_train,
              batch_size=args.batch_size,
              epochs=args.n_epochs,
              validation_data=(x_eval_encoded, y_eval),
              callbacks=[metrics]
              )
    #score, acc, f1 = model.evaluate(x_eval_encoded, y_eval,
    #                           batch_size=args.batch_size)
    #print('Test score:', score)
    #print('Test acc:', acc)
    #print('Test f1:', f1)
    #eval_pred = model.predict(x_eval_encoded)



    print('done')