import tfpyth
import torch
import torch.nn as nn
from keras import Input
from keras.layers import Bidirectional, LSTM, Dense
from pytorch_pretrained_bert import BertModel
import tensorflow as tf

session = tf.Session()

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False, explainable=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.top_rnns=top_rnns
        self.explainable = explainable
        if self.explainable:
            if self.top_rnns:
                self.rnn = lambda x: Bidirectional(LSTM(units=768 // 2, return_sequences=True))(Bidirectional(LSTM(units=768 // 2, return_sequences=True))(x))
            self.fc = Dense(vocab_size)
        else:
            if top_rnns:
                self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
            self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning

    def init_tf(self, maxlen):#, session):
        #self.tf_session = tf.Session()
        tf_inputs = Input(shape=(maxlen, 768))
        tf_model = tf_inputs
        if self.top_rnns:
            tf_model = self.rnn(tf_model)
        tf_model = self.fc(tf_model)

        #self.tf_session = tf.Session()
        session.run(tf.global_variables_initializer())

        self.tf_model = tfpyth.torch_from_tensorflow(session, [tf_inputs, ], tf_model).apply


    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.explainable:
            # move to cpu because tfpyth does not support gpu training
            #enc = enc.to('cpu')
            enc = torch.tensor(enc, requires_grad=True).to('cpu')
            logits = self.tf_model(enc)
        else:
            if self.top_rnns:
                enc, _ = self.rnn(enc)
            logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

