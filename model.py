from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Reshape, Flatten, concatenate, Bidirectional, TimeDistributed, RepeatVector, Dropout
#from keras.layers.recurrent import CuDNNLSTM as LSTM
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers

import json
import sys
import numpy as np


from keras.callbacks import Callback

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class GenerationCallback(Callback):

    def __init__(self, vocabulary):

        self.vocabulary=vocabulary
        self.inversed_vocabulary={value:key for key, value in vocabulary.items()}

    def on_epoch_end(self, epoch, logs={}):

        pass
        
       



class GenerationModel(object):

    def __init__(self, vocab_size, sequence_len, args):
        """args: parameters from argparse"""

        print("Building model",file=sys.stderr)

        # inputs
        input_=Input(shape=(sequence_len,))
        embeddings=Embedding(vocab_size, args.embedding_size)(input_)
        lstm=LSTM(args.recurrent_size, return_sequences=True)(embeddings)
        drop=Dropout(args.dropout)(lstm)
        hidden=TimeDistributed(Dense(args.hidden_dim, activation="tanh"))(drop)
        classification=TimeDistributed(Dense(vocab_size, activation="softmax"))(hidden)

        self.model=Model(inputs=[input_], outputs=[classification])

        optimizer = optimizers.Adam(lr=args.learning_rate)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)

        print(self.model.summary())
        
    
    
    
def load_model(model_fname, weight_fname):

    with open(model_fname, "rt", encoding="utf-8") as f:
        model=model_from_json(f.read())
    model.load_weights(weight_fname)

    return model

def save_model(model, model_fname, weight_fname="None", save_weights=False):

    model_json = model.to_json()
    with open(model_fname, "w") as f:
        print(model_json,file=f)
    if save_weights:
        model.save_weights(weight_fname)

if __name__=="__main__":

    pass


