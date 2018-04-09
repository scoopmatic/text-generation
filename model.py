from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Reshape, Flatten, concatenate, Bidirectional, TimeDistributed, RepeatVector
from keras.layers.recurrent import CuDNNLSTM as LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers

import json



class GenerationModel(object):

    def __init__(vocab_size, args):
        """args: parameters from argparse"""

        print("Building model",file=sys.stderr)

        # inputs
        input_=Input(shape=(sequence_len,))
        embeddings=Embedding(vocab_size, args.embedding_size)(input_)
        lstm=LSTM(args.recurrent_size, return_sequences=True)(embeddings)
        drop=Dropout(args.dropout)(lstm1)
        hidden=TimeDistributed(Dense(args.hidden_dim, activation="tanh"))(drop)
        classification=TimeDistributed(Dense(vocab_size, activation="softmax"))(hidden)

        model=Model(inputs=[input_], outputs=[classification])

        optimizer = Adam(lr=args.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)

        print(model.summary())
        
        return model
    
    
    
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

    # test
    pass


