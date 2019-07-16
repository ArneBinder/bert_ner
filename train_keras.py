import argparse

from keras import Sequential, Input, Model
from keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from keras.utils import plot_model, to_categorical
from keras_preprocessing import sequence
from pytorch_pretrained_bert import BertModel

from data_load import NerDataset, VOCAB

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--explainable", dest="explainable", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset", type=str, default="conll2003/train.txt")
    parser.add_argument("--validset", type=str, default="conll2003/valid.txt")
    args = parser.parse_args()

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading data...')
    train_dataset = NerDataset(args.trainset)
    eval_dataset = NerDataset(args.validset)
    print('train_dataset.maxlen: %i' % train_dataset.maxlen)
    print('eval_dataset.maxlen: %i' % eval_dataset.maxlen)
    maxlen = max(train_dataset.maxlen, eval_dataset.maxlen)
    assert train_dataset.vocab_size() == eval_dataset.vocab_size(), 'train_dataset.vocab_size [%i] != eval_dataset.vocab_size [%i]' % (train_dataset.vocab_size(), eval_dataset.vocab_size())
    vocab_size = train_dataset.vocab_size()

    x_train, y_train = train_dataset.get_xy()
    x_eval, y_eval = eval_dataset.get_xy()
    xy_traineval = [x_train, y_train, x_eval, y_eval]
    for i, data in enumerate(xy_traineval):
        xy_traineval[i] = sequence.pad_sequences(data, maxlen=maxlen, padding='post')
        print('data shape:', xy_traineval[i].shape)
    x_train, y_train, x_eval, y_eval = xy_traineval
    y_train = to_categorical(y_train, num_classes=len(VOCAB))
    y_eval = to_categorical(y_eval, num_classes=len(VOCAB))

    #print('Embed sequences with BERT...')
    #bert = BertModel.from_pretrained('bert-base-cased')

    print('Build model...')
    input_words = Input(shape=(maxlen, ), dtype='int32', name='word_ids')
    X = Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)(input_words)
    if args.top_rnns:
        X = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(X)
        X = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(X)
    #model.add(TimeDistributed(Dense(len(VOCAB), activation='softmax')))
    #model.add(Dense(len(VOCAB), activation='softmax'))
    pred = Dense(len(VOCAB), activation='softmax')(X)
    model = Model(input_words, pred)
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)


    print('Train with batch_size=%i...' % args.batch_size)
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.n_epochs,
              validation_data=(x_eval, y_eval)
              )
    score, acc = model.evaluate(x_eval, y_eval,
                               batch_size=args.batch_size)
    print('Test score:', score)
    print('Test acc:', acc)

    print('done')