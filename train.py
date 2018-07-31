import inp
import itertools
import json
import os
import numpy as np
import math
from keras.preprocessing.sequence import pad_sequences
from model import GenerationModel, save_model, load_model, GenerationCallback
from keras.callbacks import ModelCheckpoint
from keras.models import load_model as load_model_
import sentencepiece as spm

DATA_PATH="/home/samuel/data-textgen/all_stt.conllu.gz"

ID,FORM,LEMMA,FEATS,UPOS,XPOS,HEAD,DEPREL,DEPS,MISC=range(10)
def build_vocabularies(documents):
    char_vocab={"<PADDING>":0,"<OOV>":1,"<BOS>":2,"<EOS>":3,"<BOD>":4,"<EOD>":5,"<BOW>":6,"<EOW>":7}
    for document,meta in documents:
        for comment,sent in document:
            for cols in sent:
                for char in cols[FORM]:
                    char_vocab.setdefault(char,len(char_vocab))
    return char_vocab


def vectorize_doc(document,char_vocab):
    doc_char_vectorized=[]
    for comment,sent in document:
        sent_char_vectorized=[]
        for cols in sent:
            sent_char_vectorized.append(list(char_vocab.get(char,1) for char in cols[FORM]))
        doc_char_vectorized.append(sent_char_vectorized)
    return doc_char_vectorized


def infinite_data_vectorizer(vocabulary, data_file):
    pass


def infinite_datareader(fname, max_documents=0):
    document_counter=0
    iteration=0
    while True:
        iteration+=1
        print("Iteration:", iteration)
        for document, meta in inp.get_documents(fname):
            document_counter+=1
            yield document, iteration, document_counter-1
            if max_documents!=0 and document_counter>=max_documents:
                break

def infinite_vectorizer(vocabulary, fname, batch_size, sequence_len, sp_model=None):
    """ ... """

    inputs=np.zeros((batch_size, sequence_len)) # 0==<unk> (padding)
    outputs=np.zeros((batch_size, sequence_len))
    batch_i = 0
    for document, iteration, doc_id in infinite_datareader(fname):
        # Warning: doc_id is a counter that doesn't reset at next iteration
        # vectorize_doc: document is a list of sentences, sentence is a list of words
        #  ... and word is a list of characters
        # --> flatten this to get a document as a list of characters, and add an end-of-word
        #  ... markers to represent white space

        doc = []
        for comment,sent in document:
            sent_str = ' '.join([cols[FORM] for cols in sent])
            sent_str =  sent_str.replace(' .', '.')\
                                .replace(' ,', ',')\
                                .replace(' :', ':')
            doc.append(sent_str)

        doc_str = ' '.join(doc)
        #doc_pcs = sp.EncodeAsPieces(doc_str)
        doc_pc_ids = [sp_model.PieceToId('<s>')]+sp_model.EncodeAsIds(doc_str)+[sp_model.PieceToId('</s>')]

        #th = np.random.random()*0.5*(1-doc_id%10000/10000)+0.2
        ##th = 1-doc_id/2000000
        ##hint_ids = sp_model.EncodeAsIds(' '.join([w for w in doc if th>np.random.random()][:50]))

        ##print(" doc id: %d (%d pcs)    " % (doc_id, len(doc_pc_ids)), end="")
        data = doc_pc_ids

        #doc_input = np.zeros((batch_size, 1))
        #doc_input[:] = doc_id

        while len(data)>sequence_len:
            inputs[batch_i,:] = data[:sequence_len]
            outputs[batch_i,:] = data[1:sequence_len+1]
            data = data[sequence_len:]
        else:
            try:
                inputs[batch_i,-len(data)+1:] = data[:-1]
            except ValueError:
                if len(data) == 1: # Skip <s>-only sequence
                    continue
                else:
                    raise

            outputs[batch_i,-len(data)+1:] = data[1:]
            ##yield ([inputs[:batch_i+1,:], inputs[:batch_i+1,:]], np.expand_dims(outputs[:batch_i+1,:], -1))#np.expand_dims(outputs,-1))

        if batch_i == batch_size-1:
            yield ([inputs, inputs], np.expand_dims(outputs,-1))
            inputs=np.zeros((batch_size, sequence_len))
            outputs=np.zeros((batch_size, sequence_len))
            batch_i = 0
        else:
            batch_i += 1


#if __name__=="__main__":
import argparse
parser = argparse.ArgumentParser(description='')
g=parser.add_argument_group("Reguired arguments")

g.add_argument('--embedding_size', type=int, default=100, help='Character embedding size')
g.add_argument('--recurrent_size', type=int, default=200, help='Recurrent layer size')
g.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
g.add_argument('--hidden_dim', type=int, default=100, help='Size of the hidden layer in timedistributed')
g.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

args = parser.parse_args()
#model = train(args)


#def train(args):

sp = spm.SentencePieceProcessor()
sp.Load("m8k.sentpc")
sp_vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
inv_sp_vocab = {value:key for key, value in sp_vocab.items()}

batch_size=10
sequence_len=100#250#1601
#n_docs = 1063180

train_vectorizer=infinite_vectorizer(sp_vocab, DATA_PATH, batch_size, sequence_len, sp_model=sp)

examples = [next(train_vectorizer) for i in range(10)]
example_input = [np.concatenate([inp[0] for inp, outp in examples]), np.concatenate([inp[1] for inp, outp in examples])]
example_output = np.concatenate([outp for inp, outp in examples])

generation_model=GenerationModel(len(sp_vocab), sequence_len, args)

#from keras.utils import multi_gpu_model
#para_model=multi_gpu_model(generation_model.model, gpus=4)

generate_callback=GenerationCallback(sp_vocab, sequence_len, [generation_model, generation_model.encoder_model, generation_model.decoder_model], sp_model=sp, generator=train_vectorizer)
checkpointer = ModelCheckpoint(filepath='seq2seq_weights_6.hdf5', verbose=1, save_best_only=True, save_weights_only=True)


#generation_model.model.load_weights('seq2seq_weights_5_flat_batch.hdf5')

"""
generation_model.model.layers[2] = tmp_model.model.layers[2]
# encoder
#for trg, src in [(1,2),(2,3)]:
for trg, src in [(1,2)]:
    generation_model.encoder_model.layers[trg] = tmp_model.model.layers[trg]

# decoder
#for trg, src in [(1,2),(4,4),(6,6),(8,8)]:
for trg, src in [(1,2)]:
    generation_model.decoder_model.layers[trg] = tmp_model.model.layers[trg]
"""
"""
for layer in [5, 9, 11]:
    generation_model.model.layers[layer].set_weights(pretrained_model.layers[layer].get_weights())
layer = 7
new_weights = np.zeros((300,1200))+0.0001
new_weights[100:,:] = pretrained_model.layers[layer].get_weights()[0]
all_weights = pretrained_model.layers[layer].get_weights()
all_weights[0] = new_weights
generation_model.model.layers[layer].set_weights(all_weights)
"""
#generation_model.model.layers[2].set_weights([np.load("../neuraltopics/doc2topic/all_stt_docvecs.npy")])

# steps_per_epoch: how many batcher before running callbacks
# epochs: how many steps to run, should be basically infinite number (until killed)
#para_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
#para_model.fit_generator(train_vectorizer, steps_per_epoch=2000, epochs=1000000, verbose=1, callbacks=[generate_callback, checkpointer], validation_data=(example_input,example_output))

generation_model.model.fit_generator(train_vectorizer, steps_per_epoch=5000, initial_epoch=0, epochs=2000, verbose=1, callbacks=[generate_callback,checkpointer], validation_data=(example_input,example_output))


#return generation_model
