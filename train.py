import inp
import itertools
import json
import os
import numpy as np
import math
from keras.preprocessing.sequence import pad_sequences
from model import GenerationModel, save_model, load_model, GenerationCallback



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
            yield document, iteration
            if max_documents!=0 and document_counter>=max_documents:
                break

def infinite_vectorizer(vocabulary, fname, batch_size, sequence_len):
    """ ... """
    data=[]
    for document, iteration in infinite_datareader(fname):
        # vectorize_doc: document is a list of sentences, sentence is a list of words
        #  ... and word is a list of characters
        # --> flatten this to get a document as a list of characters, and add an end-of-word
        #  ... markers to represent white space
        vectorized_document=[vocabulary["<BOD>"]]
        for sent in vectorize_doc(document, vocabulary):
            for word in sent:
                for char in word:
                    vectorized_document.append(char)
                vectorized_document.append(vocabulary["<EOW>"])
        vectorized_document.append(vocabulary["<EOD>"])
        data+=vectorized_document
        if len(data)>batch_size*sequence_len:
            batch_input=np.array(data[:batch_size*sequence_len]).reshape(batch_size, sequence_len) # TODO do not remove the last (uncomplete) row
            # targets: shift by one character
            batch_output=np.array(data[1:batch_size*sequence_len+1]).reshape(batch_size, sequence_len)
            yield (batch_input, np.expand_dims(batch_output,-1))
            data=[]
        
 


def train(args):

    # vocabulary
    if not os.path.exists("char_vocab.json"):
        docs=inp.get_documents("/home/ginter/text-generation/all_stt.conllu.gz")
        char_vocab=build_vocabularies(itertools.islice(docs,10000))
        with open("char_vocab.json","wt") as f:
            json.dump(char_vocab,f)
    else:
        with open("char_vocab.json","rt") as f:
            char_vocab=json.load(f)

    inversed_vocabulary={value:key for key, value in char_vocab.items()}

    batch_size=60
    sequence_len=101
    train_vectorizer=infinite_vectorizer(char_vocab, "/home/ginter/text-generation/all_stt.conllu.gz", batch_size, sequence_len)

    example_input, example_output=next(train_vectorizer)
    print("input:",[inversed_vocabulary[t] for t in example_input[0,:]])
    print("output:",[inversed_vocabulary[t] for t in example_output.reshape(batch_size,sequence_len)[0,:]])

    generation_model=GenerationModel(len(char_vocab), sequence_len, args)

    generate_callback=GenerationCallback(char_vocab)

    # steps_per_epoch: how many batcher before running callbacks
    # epochs: how many steps to run, should be basically infinite number (until killed)
    generation_model.model.fit_generator(train_vectorizer, steps_per_epoch=100, epochs=1000000, verbose=1, validation_data=(example_input,example_output), callbacks=[generate_callback])
    





if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--embedding_size', type=int, default=100, help='Character embedding size')
    g.add_argument('--recurrent_size', type=int, default=200, help='Recurrent layer size')
    g.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    g.add_argument('--hidden_dim', type=int, default=100, help='Size of the hidden layer in timedistributed')
    g.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
        
    train(args)
