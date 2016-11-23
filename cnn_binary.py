# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:53:20 2016

@author: aitor
"""
import sys

from gensim.models import Word2Vec
from keras.layers import Convolution1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Input, merge
from keras.models import Model

# Base path for the Stanford Sentiment Treebank dataset
TREEBANK_PATH = './dataset/stanfordSentimentTreebank/'
TREEBANK_X_TRAIN = TREEBANK_PATH + 'x_train.json'
TREEBANK_Y_TRAIN = TREEBANK_PATH + 'y_train.json'
TREEBANK_X_DEV = TREEBANK_PATH + 'x_dev.json'
TREEBANK_Y_DEV = TREEBANK_PATH + 'y_dev.json'
TREEBANK_X_TEST = TREEBANK_PATH + 'x_test.json'
TREEBANK_Y_TEST = TREEBANK_PATH + 'y_test.json'
TREEBANK_X_PHRASES = TREEBANK_PATH + 'x_phrases.json'
TREEBANK_Y_PHRASES = TREEBANK_PATH + 'y_phrases.json'
# Word2Vec binary file not included in the github repository, download it from
# from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
WORD2VEC_DIR = './word2vec/GoogleNews-vectors-negative300.bin'
# Maximun lenght for the sentences/phrases of the dataset
MAX_PHRASE_LENGTH = 55
# Training configutation
EMBEDDING_TRAINABLE = False
DROPOUT_VALUE = 0.5
BATCH_SIZE = 20
NUMBER_OF_EPOCHS = 1500
TOTAL_CATEGORIES = 1


def _load_word2vec_model(model_dir):
    if 'bin' in model_dir:
        model = Word2Vec.load_word2vec_format(model_dir, binary=True)
    else:
        model = Word2Vec.load(model_dir)
    return model
    
def _initialize_embedding_layer():
    weights = _generate_embedding_matrix()
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights], input_length=MAX_PHRASE_LENGTH, trainable=EMBEDDING_TRAINABLE)
    return layer
    
def _generate_embedding_matrix():
    return True
    
def _load_training_data():
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    
    return x_train, y_train, x_dev, y_dev
    

# input    
input_layer = Input(shape=(MAX_PHRASE_LENGTH,), dtype='int32', name='input')    
initialized_embedding = _initialize_embedding_layer()
embedding_layer = initialized_embedding(input_layer)
# ngram convolutions
conv_3gram_layer = Convolution1D(100, 3, activation='relu', input_length=MAX_PHRASE_LENGTH)(embedding_layer)
gmp3_layer = GlobalMaxPooling1D()(conv_3gram_layer)
conv4_4gram_layer = Convolution1D(100, 4, activation='relu', input_length=MAX_PHRASE_LENGTH)(embedding_layer)
gmp4_layer = GlobalMaxPooling1D()(conv4_4gram_layer)
conv5_5gram_layer = Convolution1D(100, 5, activation='relu', input_length=MAX_PHRASE_LENGTH)(embedding_layer)
gmp5_layer = GlobalMaxPooling1D()(conv5_5gram_layer)
ngrams_merge_layer = merge([gmp3_layer, gmp4_layer, gmp5_layer], mode='concat')
#output
dropout_layer = Dropout(DROPOUT_VALUE)(ngrams_merge_layer)
softmax_layer = Dense(TOTAL_CATEGORIES, activation="softmax")(dropout_layer)
#model
model = Model(input=[input_layer], output=[softmax_layer])
# We are classiyinf the phrases/sentences on 2 categories with the specialized models
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
model.summary()
x_train, y_train, x_dev, y_dev = _load_training_data()
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUMBER_OF_EPOCHS, validation_data=(x_dev, y_dev))
print 'Done'
    
    
    