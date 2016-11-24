# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:53:20 2016

@author: aitor
"""

#from exceptions import KeyError
import json
import sys

from gensim.models import Word2Vec
from keras.layers import Convolution1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Input, merge
from keras.models import Model
import numpy as np
#from mytext import Tokenizer
from keras.preprocessing.text import Tokenizer

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
# Number of dimensions of each word vector
WORD_DIMENSIONS = 300

# Sentiments
VERY_NEGATIVE = 'very_negative'
NEGATIVE = 'negative'
NEUTRAL = 'neutral'
POSITIVE = 'positive'
VERY_POSITIVE = 'very_positive'
SENTIMENT_VECTOR = [VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE]

# Training configutation
EMBEDDING_TRAINABLE = False
DROPOUT_VALUE = 0.5
BATCH_SIZE = 20
NUMBER_OF_EPOCHS = 1500
TOTAL_CATEGORIES = 1

# Load the precalculated word2vec model
def _load_word2vec_model(model_dir):
    if 'bin' in model_dir:
        model = Word2Vec.load_word2vec_format(model_dir, binary=True)
    else:
        model = Word2Vec.load(model_dir)
    return model

# Initialize the embedding layer using the weight of the precalculated word2vec
# embeddings.    
def _initialize_embedding_layer():
    weights = _generate_embedding_matrix()
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights], input_length=MAX_PHRASE_LENGTH, trainable=EMBEDDING_TRAINABLE)
    return layer
    
def _generate_embedding_matrix():
    all_sentences = _get_all_sentences()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    word_index = tokenizer.word_index
    model = _load_word2vec_model(WORD2VEC_DIR)
    embedding_matrix = np.zeros((len(word_index) + 1, WORD_DIMENSIONS))
    unknown_words = {}
    for word, i in word_index.items():
        try:
            embedding_vector = model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            if word in unknown_words:
                unknown_words[word] += 1
            else:
                unknown_words[word] = 1
    
    return embedding_matrix

# get all sentences in the dataset    
def _get_all_sentences():
    x_train = json.load(open(TREEBANK_X_TRAIN, 'r'))
    x_dev = json.load(open(TREEBANK_X_DEV, 'r'))
    x_test = json.load(open(TREEBANK_X_TEST, 'r'))
    all_sentences = x_train + x_dev + x_test
    return all_sentences
    
# Load the data used during the training (train + dev)    
def _load_training_data(target_sentiment, use_phrases = True):
    x_train = json.load(open(TREEBANK_X_TRAIN, 'r'))
    y_train = json.load(open(TREEBANK_Y_TRAIN, 'r'))
    if use_phrases:
        x_phrases = json.load(open(TREEBANK_X_PHRASES, 'r'))
        y_phrases = json.load(open(TREEBANK_Y_PHRASES, 'r'))
        x_train = x_train + x_phrases
        y_train = y_train + y_phrases  
    x_dev = json.load(open(TREEBANK_X_DEV, 'r'))
    y_dev = json.load(open(TREEBANK_Y_DEV, 'r')) 
    
    all_sentences = _get_all_sentences()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences)
#    sequences = tokenizer.texts_to_sequences(all_sentences_X)
    sequences_x_train = tokenizer.texts_to_sequences(x_train)
    sequences_x_dev = tokenizer.texts_to_sequences(x_dev)
    
    sentiment_index = SENTIMENT_VECTOR.index(target_sentiment)
    sequences_y_train = []
    for example in y_train:
        sequences_y_train.append(example[sentiment_index])        
    sequences_y_dev = []    
    for example in y_dev:
        sequences_y_dev.append(example[sentiment_index])    
    
    return sequences_x_train, sequences_y_train, sequences_x_dev, sequences_y_dev

# Load the data used during the testing (test)    
def _load_test_data():
    x_test = json.load(open(TREEBANK_X_TEST, 'r'))
    y_test = json.load(open(TREEBANK_Y_TEST, 'r'))
    return x_test, y_test
    
    
print 'Starting...'
sys.stdout.flush()  
print 'Initializing embedding layer...'
sys.stdout.flush()
# initialize the embedding layer using teh pretrained word2vec
initialized_embedding = _initialize_embedding_layer()
print 'Building model...'
sys.stdout.flush()
input_layer = Input(shape=(MAX_PHRASE_LENGTH,), dtype='int32', name='input') 
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
# We are classifying the phrases/sentences on 2 categories with the specialized models
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
model.summary()
print 'Loading training data...'
sys.stdout.flush()
x_train, y_train, x_dev, y_dev = _load_training_data(VERY_NEGATIVE) # set the target sentiment
print 'Training...'
sys.stdout.flush()
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUMBER_OF_EPOCHS, validation_data=(x_dev, y_dev))
print 'Done'
    
    
    