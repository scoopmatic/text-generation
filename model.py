from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Reshape, Flatten, concatenate, Bidirectional, TimeDistributed, RepeatVector, Dropout, InputSpec
from keras.layers import CuDNNLSTM as LSTM
#from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.preprocessing import sequence
import keras.backend as K

import json
import sys
import numpy as np
import random
import heapq

alpha = 0.975
#len_norm = (lambda x: x[0]/((5+len(x[1]))**alpha/(5+1)**alpha))
len_norm = (lambda x: x[0]/len(x[1])**alpha)
vocab_norm = (lambda x: x[0]/len(set(x[1]))**alpha)

from keras.callbacks import Callback

def sample_token(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def input_sequence(sample, seq_len):
	sequence = np.zeros((1,seq_len)) # <unk>s
	sequence[0,0:len(sample)] = sample
	return sequence


def mrbeam(state, decoder, k=3, max_len=100, seq_len=400, sp_model=None):
	""" Beam search text generator """
	UNK, BOS, EOS = 0, 1, 2
	dead_samples = []
	dead_scores = []
	live_samples = [[BOS]]
	live_scores = [0]
	for i in range(min(seq_len, max_len)):
		print("\ri: %d, alive: %d, dead: %d   " % (i, len(live_samples), len(dead_samples)), end="")
		live_sequences = np.concatenate([input_sequence(sample, seq_len) for sample in live_samples])
		rep = live_sequences.shape[0]
		state_ = [np.repeat(state[0],rep,axis=0).reshape((rep,state[0].shape[1])), np.repeat(state[1],rep,axis=0).reshape((rep,state[1].shape[1]))]
		pred, _, _ = decoder.predict([live_sequences] + state_)

		tops = [np.argsort(pred[n][i])[-k:] for n in range(pred.shape[0])]
		probs = [[pred[n][i][j] for j in tops[n]] for n in range(pred.shape[0])]

		spare_samples, spare_scores = [], []
		#print()
		for n, prob_dist in enumerate(probs):
			#print(n, sp_model.DecodeIds(live_samples[n]), ':', ' '.join([sp_model.IdToPiece(int(id))+("(%.3f)"%probs[n][i]) for i, id in enumerate(tops[n])]))
			"""
			# Experimental: multinomial sampling over token probability distribution
			exp_probs = np.exp(np.log(prob_dist) / 0.6)
			selected = False
			while not selected:
				picks = [np.argmax(np.random.multinomial(1, exp_probs / np.sum(exp_probs), 1)) for x in range(10)]
				picks = [(p, prob_dist[p]) for p in set(picks)]
				for m, token_prob in picks:"""
			for m, token_prob in enumerate(prob_dist):
				samples = live_samples[n] + [int(tops[n][m])]
				scores = live_scores[n] - np.log(token_prob)
				if tops[n][m] == EOS:
					dead_samples.append(samples + [sp_model.PieceToId('/')]*2)
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

		try:
			live_scores, live_samples = zip(*heapq.nsmallest(k, zip(spare_scores, spare_samples), key=vocab_norm))
		except:
			import pdb; pdb.set_trace()
		if len(live_samples) == 0:
			break
	print()
	return sorted(zip(dead_scores + list(live_scores), dead_samples + list(live_samples)), key=vocab_norm, reverse=True)


class TimestepDropout(Dropout):
	"""Timestep Dropout.

	This version performs the same function as Dropout, however it drops
	entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

	# Arguments
		rate: float between 0 and 1. Fraction of the timesteps to drop.

	# Input shape
		3D tensor with shape:
		`(samples, timesteps, channels)`

	# Output shape
		Same as input

	# References
		- A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
		- https://github.com/keras-team/keras/issues/7290
	"""

	def __init__(self, rate, **kwargs):
		super(TimestepDropout, self).__init__(rate, **kwargs)
		self.input_spec = InputSpec(ndim=3)

	def _get_noise_shape(self, inputs):
		input_shape = K.shape(inputs)
		noise_shape = (input_shape[0], input_shape[1], 1)
		return noise_shape


class GenerationCallback(Callback):

	def __init__(self, vocabulary, seq_len, models, sp_model=None, generator=None, val_input=None):

		self.vocabulary=vocabulary
		self.inverse_vocabulary={value:key for key, value in vocabulary.items()}
		self.model, self.encoder_model, self.decoder_model = models
		self.seq_len = seq_len
		self.sp_model = sp_model
		self.generator = generator
		self.val_input = val_input

	def on_epoch_end(self, epoch, logs={}):
		vocab = self.vocabulary
		inv_vocab = self.inverse_vocabulary
		try:
			input, output = next(self.generator)
			encoder_in, decoder_in = input
		except ValueError as err:
			print(err)
			return

		for enc_in in [self.val_input[0][0:1,:], encoder_in]:
			state = self.encoder_model.predict(enc_in[0:1,:])
			print(self.sp_model.DecodeIds([int(x) for x in enc_in[0,] if x > 0]))
			beams = mrbeam(state, self.decoder_model, k=30, max_len=epoch//4+20, seq_len=self.seq_len, sp_model=self.sp_model)
			for n, (score, sample) in enumerate(beams):
				if n % (len(beams)//8) == 0 or n >= len(beams)-5:
					print("%d --> %s (%.3f)" % (n, self.sp_model.DecodeIds([int(x) for x in sample]), vocab_norm((score, sample))))


		# Multinomial sampling over token probability distribution
		for diversity in [0.35]:#[0.2, 0.5]:
			unk_cnt = 0
			print("Diversity: %.1f" % diversity)
			generated = []
			#sys.stdout.write(" ".join(generated))
			generate_X = np.zeros((1,self.seq_len))
			generate_X[:,0] = self.sp_model.PieceToId('<s>')
			for i in range(0, 48):
				# predict
				#preds = self.model.predict([doc_id, generate_X], verbose=0)[0]
				#preds = self.model.predict(generate_X, verbose=0)[0]
				preds = self.decoder_model.predict([generate_X] + state, verbose=0)[0]
				next_index = sample_token(preds[0,i,:], diversity)
				next_char = inv_vocab[next_index]
				generate_X[:,i+1] = next_index
				#generate_X = np.array([np.append(generate_X, [next_index])])

				#sentence=generated[len(generated)-context_size:]

				# vectorize new seed
				#generate_X=np.zeros((1,context_size))
				#for i,c in enumerate(sentence):
				#    generate_X[0,i]=vocab.get(c,vocab["<UNKNOWN>"])

				sys.stdout.write(' ')
				sys.stdout.write(self.sp_model.IdToPiece(int(next_index)))
				if next_index in [self.sp_model.PieceToId('</s>'), self.sp_model.PieceToId('<unk>')]:
					out = self.sp_model.IdToPiece(int(next_index))
					##print(out, end="")
					unk_cnt += 1
					if unk_cnt > 3 or out == '</s>':
						print()
						break

				sys.stdout.flush()

			sys.stdout.write("\n")
			sys.stdout.flush()
			print("%s\n --> %s" % (self.sp_model.DecodeIds([int(x) for x in encoder_in[0,] if x > 0]),
								self.sp_model.DecodeIds([int(x) for x in generate_X[0,:i]])))


class GenerationModel(object):

	def __init__(self, vocab_size, sequence_len, args):
		"""args: parameters from argparse"""
		deep = False
		print("Building model",file=sys.stderr)
		## Encoder/decoder embeddings
		shared_embeddings = Embedding(vocab_size, args.embedding_size, name="embeddings")

		## Encoder model
		encoder_input = Input(shape=(sequence_len,))
		encoder_emb = shared_embeddings(encoder_input)

		encoder_lstm1 = LSTM(args.recurrent_size, return_sequences=False, return_state=True, name="encoder_lstm1")(encoder_emb)
		encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm1
		encoder_outputs = Dropout(args.dropout)(encoder_outputs)
		#encoder_outputs, encoder_state_h, encoder_state_c = encoder_drop1
		encoder_states = [encoder_state_h, encoder_state_c]

		## Decoder model (for training)
		decoder_input = Input(shape=(sequence_len,)) # Gold standard input for training
		decoder_emb = shared_embeddings(decoder_input)
		##decoder_emb = TimestepDropout(0.5)(decoder_emb)
		# Embeddings and state as input (peeky)
		encoder_state_h_rep = RepeatVector(sequence_len)(encoder_state_h)
		encoder_state_c_rep = RepeatVector(sequence_len)(encoder_state_c)
		decoder_merged_input = concatenate([decoder_emb, encoder_state_h_rep, encoder_state_c_rep])

		"""# Begin decoder LSTM block
		decoder_lstm1 = LSTM(args.recurrent_size, return_sequences=True, name="decoder_lstm1")
		decoder_drop1 = Dropout(args.dropout)(decoder_lstm1)
		decoder_lstm2 = LSTM(args.recurrent_size, return_sequences=True, return_state=True, name="decoder_lstm2")(decoder_drop1, initial_state=encoder_states)
		decoder_lstm_block = Dropout(args.dropout)(decoder_lstm2) # End block"""

		decoder_lstm1 = LSTM(args.recurrent_size, return_sequences=True, return_state=True, name="decoder_lstm1")
		decoder_lstm2 = LSTM(args.recurrent_size, return_sequences=True, return_state=True, name="decoder_lstm2")
		hidden=TimeDistributed(Dense(args.hidden_dim, activation="tanh"))
		decoder_softmax = TimeDistributed(Dense(vocab_size, activation="softmax", name="decoder_out"))
		#decoder_output, _, _ = decoder_lstm_block(decoder_emb, initial_state=encoder_states)

		decoder_lstm1_output, _, _ = decoder_lstm1(decoder_merged_input, initial_state=encoder_states)
		#decoder_lstm1_output, _, _ = decoder_lstm1(decoder_emb, initial_state=encoder_states)

		decoder_lstm1_drop = Dropout(args.dropout)(decoder_lstm1_output)

		decoder_lstm2_output, _, _ = decoder_lstm2(decoder_lstm1_drop, initial_state=encoder_states)
		decoder_lstm2_drop = Dropout(args.dropout)(decoder_lstm2_output)

		if not deep:
			decoder_output = hidden(decoder_lstm1_drop)
			decoder_output = Dropout(args.dropout)(decoder_output)
			decoder_output = decoder_softmax(decoder_output)
		else:
			decoder_output = decoder_softmax(decoder_lstm2_drop)

		self.model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])
		self.encoder_model = Model(encoder_input, encoder_states) # Stand-alone encoder

		## Stand-Alone decoder model (for generation)
		SA_decoder_state_input_h = Input(shape=(args.recurrent_size,))
		SA_decoder_state_input_c = Input(shape=(args.recurrent_size,))
		SA_decoder_state_input = [SA_decoder_state_input_h, SA_decoder_state_input_c]

		# State as input variable (as well as initial state), merged with embeddings (of generated sequence)
		SA_decoder_state_input_h_rep = RepeatVector(sequence_len)(SA_decoder_state_input_h)
		SA_decoder_state_input_c_rep = RepeatVector(sequence_len)(SA_decoder_state_input_c)
		SA_decoder_merged_input = concatenate([decoder_emb, SA_decoder_state_input_h_rep, SA_decoder_state_input_c_rep])

		#decoder_output, state_h, state_c = decoder_lstm_block(decoder_emb, initial_state=decoder_state_input)

		SA_decoder_lstm1_output, state_h, state_c = decoder_lstm1(SA_decoder_merged_input, initial_state=SA_decoder_state_input)
		##SA_decoder_lstm1_output, _, _ = decoder_lstm1(SA_decoder_merged_input, initial_state=SA_decoder_state_input)
		#SA_decoder_lstm1_output, _, _ = decoder_lstm1(decoder_emb, initial_state=SA_decoder_state_input)

		SA_decoder_lstm1_drop = Dropout(args.dropout)(SA_decoder_lstm1_output)
		SA_decoder_lstm2_output, state_h, state_c = decoder_lstm2(SA_decoder_lstm1_drop, initial_state=SA_decoder_state_input)
		SA_decoder_lstm2_drop = Dropout(args.dropout)(SA_decoder_lstm2_output)
		if not deep:
			SA_decoder_output = hidden(SA_decoder_lstm1_drop)
			SA_decoder_output = Dropout(args.dropout)(SA_decoder_output)
			SA_decoder_output = decoder_softmax(SA_decoder_output)
		else:
			SA_decoder_output = decoder_softmax(SA_decoder_lstm2_drop)
		SA_decoder_states = [state_h, state_c]
		#decoder_output = decoder_softmax(decoder_output)
		self.decoder_model = Model([decoder_input] + SA_decoder_state_input, [SA_decoder_output] + SA_decoder_states)

		optimizer = optimizers.Adam(lr=args.learning_rate, amsgrad=True)
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
