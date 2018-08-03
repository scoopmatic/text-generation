import inp
import itertools
import json
import os
import numpy as np
import math
from keras.preprocessing.sequence import pad_sequences
from model import GenerationModel, save_model, load_model, GenerationCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model as load_model_
import keras.backend as K
import sentencepiece as spm

# Config GPU memory usage
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, clear_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = -1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


#DATA_PATH="/home/samuel/data-textgen/all_stt.conllu.gz"
DATA_PATH="/home/samuel/data/all_stt.conllu.gz"

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
            #if len(doc) >= iteration+1:
            #    break # limit doc size to 2 sents

        #doc = doc[-2:]
        for d_i in range(iteration%2, len(doc)-1,2):
            #doc_str = ' '.join(doc)
            doc_str = doc[d_i]+' '+doc[d_i+1]
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

batch_size=20
sequence_len=60#250#1601
#n_docs = 1063180

train_vectorizer=infinite_vectorizer(sp_vocab, DATA_PATH, batch_size, sequence_len, sp_model=sp)

examples = [next(train_vectorizer) for i in range(10)]
example_input = [np.concatenate([inp[0] for inp, outp in examples]), np.concatenate([inp[1] for inp, outp in examples])]
example_output = np.concatenate([outp for inp, outp in examples])

generation_model=GenerationModel(len(sp_vocab), sequence_len, args)

#from keras.utils import multi_gpu_model
#para_model=multi_gpu_model(generation_model.model, gpus=4)

generate_callback=GenerationCallback(sp_vocab, sequence_len, [generation_model, generation_model.encoder_model, generation_model.decoder_model], sp_model=sp, generator=train_vectorizer, val_input=example_input)
checkpointer = ModelCheckpoint(filepath='seq2seq_weights_7_mini2_lred.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
earlystopper = EarlyStopping(patience=5, verbose=1)
learnreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000001, verbose=1)

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

generation_model.model.load_weights('seq2seq_weights_7_mini2_lred.hdf5')
init_ep = 0
#while True:
#    print("Loading model")
hist = generation_model.model.fit_generator(train_vectorizer, steps_per_epoch=5000, initial_epoch=init_ep, epochs=2000, verbose=1, callbacks=[generate_callback,checkpointer,learnreducer], validation_data=(example_input,example_output))
#    init_ep = len(hist.history['val_loss'])
#    new_lr = K.get_value(generation_model.model.optimizer.lr)*(1-0.2)
#    K.set_value(generation_model.model.optimizer.lr, new_lr)
#    print("New lr:", new_lr)
#    generation_model.model.load_weights('seq2seq_weights_5_seq100_c.hdf5')

#return generation_model


alpha = 0.975
vocab_norm = (lambda x: x[0]/len(set(x[1]))**alpha)

"""def input_sequence(sample):
    sequence = np.zeros((1,sequence_len)) # <unk>s
    sequence[0,0:len(sample)] = sample
    return sequence"""
def input_sequence(sample, seq_len):
  sequence = np.zeros((1,seq_len)) # <unk>s
  sequence[0,0:len(sample)] = sample
  return sequence


k = 3
UNK, BOS, EOS = 0, 1, 2
dead_k = 0 # samples that reached eos
dead_samples = []
dead_scores = []
live_k = 1 # samples that did not yet reached eos
live_samples = [[BOS]]
live_scores = [0]


import heapq
#for ex in range(5):
state = generation_model.encoder_model.predict(example_input[0][ex:ex+1,:])
max_len=50
seq_len=sequence_len
decoder=generation_model.decoder_model

sp.DecodeIds(list(map(int,example_input[0][0,:])))
for i in range(min(seq_len, max_len)):
    print("\ri: %d, alive: %d, dead: %d   " % (i, len(live_samples), len(dead_samples)), end="")
    live_sequences = np.concatenate([input_sequence(sample, seq_len) for sample in live_samples])
    rep = live_sequences.shape[0]
    state_ = [np.repeat(state[0],rep,axis=0).reshape((rep,state[0].shape[1])), np.repeat(state[1],rep,axis=0).reshape((rep,state[1].shape[1]))]
    pred, _, _ = decoder.predict([live_sequences] + state_)
    #
    tops = [np.argsort(pred[n][i])[-k:] for n in range(pred.shape[0])]
    probs = [[pred[n][i][j] for j in tops[n]] for n in range(pred.shape[0])]
    #
    spare_samples, spare_scores = [], []
    #print()
    for n, prob_dist in enumerate(probs):
    	#print(n, sp_model.DecodeIds(live_samples[n]), ':', ' '.join([sp_model.IdToPiece(int(id))+("(%.3f)"%probs[n][i]) for i, id in enumerate(tops[n])]))
    	for m, token_prob in enumerate(prob_dist):
    		samples = live_samples[n] + [int(tops[n][m])]
    		scores = live_scores[n] - np.log(token_prob)
    		if tops[n][m] == EOS:
    			dead_samples.append(samples + [sp.PieceToId('/')]*2)
    			dead_scores.append(scores)
    			#selected = True
    		elif tops[n][m] == UNK:
    			pass
    			#spare_samples.append(live_samples[n])
    			#spare_scores.append(live_scores[n])
    		else:
    			spare_samples.append(samples)
    			spare_scores.append(scores)
    			#selected = True
    #
    #try:
    live_scores, live_samples = zip(*heapq.nsmallest(k, zip(spare_scores, spare_samples), key=vocab_norm))
    #except:
    #	import pdb; pdb.set_trace()
    if len(live_samples) == 0:
    	break

print("Dead:")
for n, (score, sample) in enumerate(sorted(zip(dead_scores, dead_samples), key=vocab_norm, reverse=True)):
    #if n % (len(beams)//8) == 0 or n >= len(beams)-5:
    print("%d --> %s (%.3f)" % (n, sp.DecodeIds([int(x) for x in sample]), vocab_norm((score, sample))))

print("Live:")
for n, (score, sample) in enumerate(sorted(zip(list(live_scores), list(live_samples)), key=vocab_norm, reverse=True)):
    #if n % (len(beams)//8) == 0 or n >= len(beams)-5:
    print("%d --> %s (%.3f)" % (n, sp.DecodeIds([int(x) for x in sample]), vocab_norm((score, sample))))


unk_cnt = 0
diversity=0.35
generated = []
generate_X = np.zeros((1,seq_len))
generate_X[:,0] = sp.PieceToId('<s>')
for i in range(0, 48):
    #i+=1
    #pred, _, _ = decoder.predict([live_sequences] + state_)
    preds = decoder.predict([generate_X] + state, verbose=0)[0]
    next_index = int(sample_token(preds[0,i,:], diversity))
    next_char = sp.IdToPiece(next_index)
    next_char
    generate_X[:,i+1] = next_index
    #sys.stdout.write(' ')
    #sys.stdout.write(sp.IdToPiece(int(next_index)))
    if next_index in [sp.PieceToId('</s>'), sp.PieceToId('<unk>')]:
    	out = sp.IdToPiece(int(next_index))
    	##print(out, end="")
    	unk_cnt += 1
    	#if unk_cnt > 3 or out == '</s>':
    	#	print()
    	#	break
    #
    sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()
print("%s\n --> %s" % (sp.DecodeIds([int(x) for x in example_input[0][0,:] if x > 0]),
					sp.DecodeIds([int(x) for x in generate_X[0,:i]])))
