import time
import re
import pickle
import os
from collections import defaultdict,OrderedDict
import multiprocessing
import collections
import json

from tqdm import tnrange, tqdm_notebook
from tqdm import tqdm, tqdm_pandas
tqdm_notebook().pandas()

import numpy as np
import pandas as pd
pd.options.display.max_rows = 25
pd.options.display.max_columns = 999

#from datetime import datetime, timedelta, timezone
import keras

from IPython.display import display

from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation, GRU, Embedding
from keras.layers import concatenate as Concatenate
from keras.layers.core import Flatten, Reshape
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.callbacks import Callback, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras import regularizers
from keras.initializers import *
import keras.backend as K
import subprocess
from keras.callbacks import ModelCheckpoint
from keras.layers import Cropping1D, Average, PReLU, ZeroPadding1D, Lambda, RepeatVector
from keras.regularizers import l1_l2, l1, l2
from keras import initializers

import warnings
from sklearn import exceptions
warnings.filterwarnings("ignore", category=exceptions.UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import nltk
nltk.download('punkt')


# import du training set
jsondata = []
with open('trainingset_final.json', 'r') as f:
    for line in f:
        rowdata = json.loads(line)
        jsondata += [rowdata]


df = pd.DataFrame.from_dict(jsondata)
del jsondata



# import du test set
jsondata = []
with open('testset.json', 'r') as f:
    for line in f:
        rowdata = json.loads(line)
        jsondata += [rowdata]

dftest = pd.DataFrame.from_dict(jsondata)
del jsondata


# fonction de pretraitement des tweets
def cleanText(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    words = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.replace("le pen", "lepen")
	sentence = sentence.replace(":", "")
	sentence = sentence.replace("rt", "")
        words+= nltk.tokenize.word_tokenize(sentence)
    return words

# on applique cleanText au deux set
df["txClean"] = df["tx"].progress_apply(cleanText)
dftest["txClean"] = df["tx"].progress_apply(cleanText)



wordDic = collections.defaultdict(int)

for row in df["txClean"]:
    for word in row:
        wordDic[word] += 1

wordList = sorted(wordDic.items(), reverse=True, key=lambda x:x[1] )

#keep the 5000 leading words and make basic embedding
top_words = 50
wordDic = {None:0} #0 is used for padding so associated it to None

for word,freq in wordList[:top_words-1]:
  wordDic[word] = len(wordDic)
#max number of words seen in a tweet
df["txClean"].apply(lambda x:len(x)).max()
dftest["txClean"].apply(lambda x:len(x)).max()

max_review_length = 50 #maximum number of words in a tweet

def wordsToNumbers(words):
    array = np.array([wordDic.get(word,0) for word in words])
    return np.pad(array[:max_review_length], pad_width=(0,max_review_length-array.shape[0]), mode="constant")


## Word to number pour les deux sets
df["txCleanNumber"] = df["txClean"].apply(wordsToNumbers)

dftest["txCleanNumber"] = dftest["txClean"].apply(wordsToNumbers)



# create the model
embedding_vecor_length = 64
model = Sequential()
model.add(Dense(500, input_shape=(top_words,)))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dense(60))
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

trainingFeatures = np.array(df.loc[:, "txCleanNumber"].tolist())

train_feat = []
for i in trainingFeatures:
    new_i = np.zeros((top_words))
    for j in i:
	if j != 0:
	    new_i[j] = 1
    train_feat.append(new_i)

train_feat = np.array(train_feat)
trainingLabels   = df.loc[:, "sent"].as_matrix()
trainingClasses = keras.utils.to_categorical(trainingLabels, num_classes=4)
model.fit(train_feat, trainingClasses, epochs=4, verbose=1)

# On ne test que pour sent = 1 ou 0
testingFeatures = np.array(dftest.loc[dftest["sent"]<2, "txCleanNumber"].tolist())

test_feat = []
for i in testingFeatures:
    new_i = np.zeros((top_words))
    for j in i:
	if j != 0:
	    new_i[j] = 1
    test_feat.append(new_i)

test_feat = np.array(test_feat)

testingLabels   = dftest.loc[dftest["sent"]<2, "sent"].as_matrix()
testingClasses = keras.utils.to_categorical(testingLabels, num_classes=4)

# Final evaluation of the model
scores = model.evaluate(test_feat, testingClasses, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

pickle.dump( wordDic, open( "wordDic.pkl", "wb" ) )

pred_vec = model.predict(test_feat, batch_size=512, verbose=1)

pred =[]

for vec in pred_vec:
    max_num = np.amax(vec)
    count = 0
    for val in vec:
	if val == max_num:
            pred.append(count)
        count = count + 1


dftest.loc[dftest["sent"]<2, "pred"] = pred

print dftest.loc[dftest["pred"]<2, ["sent", "pred"]]
