# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:24:37 2016

@author: aitor
"""

import json

# Base path for the Stanford Sentiment Treebank dataset
TREEBANK_PATH = './dataset/stanfordSentimentTreebank/'
TREEBANK_TRAIN = TREEBANK_PATH + 'train.json'
TREEBANK_DEV = TREEBANK_PATH + 'dev.json'
TREEBANK_TEST = TREEBANK_PATH + 'test.json'
TREEBANK_PHRASES = TREEBANK_PATH + 'phrases.json'

# Sentiments
VERY_NEGATIVE = 'very_negative'
NEGATIVE = 'negative'
NEUTRAL = 'neutral'
POSITIVE = 'positive'
VERY_POSITIVE = 'very_positive'
SENTIMENT_VECTOR = [VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE]

def _classify_sentiment_value(value):
    # Sentiment label for each sentence ID
    # [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    # very negative, negative, neutral, positive, very positive, respectively.
    sentiment = ''
    if value <= 0.2:
        sentiment = VERY_NEGATIVE
    elif value > 0.2 and value <= 0.4:
        sentiment = NEGATIVE
    elif value > 0.4 and value <= 0.6:
        sentiment = NEUTRAL
    elif value > 0.6 and value <= 0.8:
        sentiment = POSITIVE
    elif value > 0.8:
        sentiment = VERY_POSITIVE 
        
    return sentiment
    
def _get_sentiment_one_hot(sentiment_label):
    vector = [0] * len(SENTIMENT_VECTOR)
    index = SENTIMENT_VECTOR.index(sentiment_label)
    vector[index] = 1
    return vector
    
def _prepare_set(filepath):
    dataset = json.load(open(filepath, 'r'))
    x = []
    y = []
    for example in dataset:
        words = example[0].split(' ')
        sentiment = _classify_sentiment_value(example[1])
        sentiment_vector = _get_sentiment_one_hot(sentiment)
        x.append(words)
        y.append(sentiment_vector)
        
    return x, y
    
def prepare_sets():
    x_train, y_train = _prepare_set(TREEBANK_TRAIN)
    print 'Total train:', len(x_train)
    json.dump(x_train, open(TREEBANK_PATH + 'x_train.json', 'w'))
    json.dump(y_train, open(TREEBANK_PATH + 'y_train.json', 'w'))
    x_dev, y_dev = _prepare_set(TREEBANK_DEV)
    print 'Total dev:', len(x_dev)
    json.dump(x_dev, open(TREEBANK_PATH + 'x_dev.json', 'w'))
    json.dump(y_dev, open(TREEBANK_PATH + 'y_dev.json', 'w'))
    x_test, y_test = _prepare_set(TREEBANK_TEST)
    print 'Total test:', len(x_test)
    json.dump(x_test, open(TREEBANK_PATH + 'x_test.json', 'w'))
    json.dump(y_test, open(TREEBANK_PATH + 'y_test.json', 'w'))
    x_phrases, y_phrases = _prepare_set(TREEBANK_PHRASES)
    print 'Total phrases:', len(x_phrases)
    json.dump(x_phrases, open(TREEBANK_PATH + 'x_phrases.json', 'w'))
    json.dump(y_phrases, open(TREEBANK_PATH + 'y_phrases.json', 'w'))

print 'Start...'
prepare_sets()
print 'Done'

    
    
    
    
    
