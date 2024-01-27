# -*- coding: utf-8 -*-
#----Import Dependencies------

import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras import regularizers
# %load_ext tensorboard
import datetime, os
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
print('Done')

# %reload_ext tensorboard

#----Import Combined Reddit Data-----

train = pd.read_csv('/content/reddit.csv')

#----Check for NULL values----

train["comment"].isnull().sum()

#----Possibly remove NULL values---
train["comment"].fillna("No content", inplace = True)

#----Remove URLs, emails, new-line-characters, single quotes----

def depure_data(data):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    data = re.sub('\S*@\S*\s?', '', data)
    data = re.sub('\s+', ' ', data)
    data = re.sub("\'", "", data)
    data = re.sub("\"", "", data)
    data = re.sub("\[", "", data)
    data = re.sub("\]", "", data)
    data = re.sub("\(", "", data)
    data = re.sub("\)", "", data)
    data = re.sub("\,", "", data)
    return data

#----Change Combined Data-set----
temp = []
data_to_list = train['comment'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
list(temp[:5])

#----More Text-Cleaning----

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


data_words = list(sent_to_words(temp))

print(data_words[:10])

len(data_words)

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])

data = np.array(data)

print(data_words[:40])

#----Create labels----

labels = np.array(train['subreddit'])
y = []
for i in range(len(labels)):
    if labels[i] == 'Republican':
        y.append(0)
    if labels[i] == 'democrats':
        y.append(1)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 2, dtype="float32")
del y

len(labels)

#----Prepare to Split----

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
reddits = pad_sequences(sequences, maxlen=max_len)
print(reddits)

print(labels)

#----Split Data----
X_train, X_test, y_train, y_test = train_test_split(reddits,labels, random_state=0)
print (len(X_train),len(X_test),len(y_train),len(y_test))

#-----Build Model-----

model = Sequential()
model.add(layers.Embedding(max_words, 40, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(20,dropout=0.5)))
model.add(layers.Dense(2,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#----Train Model---

model1 = model.fit(X_train, y_train, epochs=70, batch_size=50, verbose=1, callbacks=[tensorboard_callback])

#---Evaluate model---

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)

print(val_loss)
print(val_acc)

#----Perform Tuning---

#-----Create confusion matrix----

predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))

import seaborn as sns
conf_matrix = pd.DataFrame(matrix, index = ['Republican', 'Democrat'],columns = ['Republican', 'democrat'])
#Normalizing
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize = (15,15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})

#----Test with Example----

party = ['Republican', 'democrat']

sequence = tokenizer.texts_to_sequences(["Philadelphia Was The Prototype For Election Fraud In 2020, And I Can Prove It"])
test = pad_sequences(sequence, maxlen=max_len)
party[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]

plt.plot(history.history['accuracy'])

#---Plot Design---

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

