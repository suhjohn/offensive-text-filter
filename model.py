import os
import pandas
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.python.keras.layers import Dropout

nltk.download('punkt')
nltk.download('stopwords')

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
USE_MAX_POOL = True
LAYER_COUNT = 1
MAX_SEQUENCE_LENGTH = 70
DIMENSION_SIZE = 200
DENSE_SIZE = 10
FILTER_NUM = 32
KERNEL_SIZE = 10
VOCAB_SIZE = 100000

result_text = []

print(f"USE_MAX_POOL = {USE_MAX_POOL}")
print(f"LAYER_COUNT = {LAYER_COUNT}")
print(f"MAX_SEQUENCE_LENGTH = {MAX_SEQUENCE_LENGTH}")
print(f"DIMENSION_SIZE = {DIMENSION_SIZE}")
print(f"DENSE_SIZE = {DENSE_SIZE}")
print(f"FILTER_NUM = {FILTER_NUM}")
print(f"KERNEL_SIZE = {KERNEL_SIZE}")

PRECOMPILED_WV_FILENAME = f"./glove.twitter.27B.{DIMENSION_SIZE}d.txt"
TRAINING_OFFENSIVE_FILENAME = "./training_offensive.csv"
TRAINING_REGULAR_FILENAME = "./training_regular.csv"
TEST_OFFENSIVE_FILENAME = "./test_offensive.csv"
TEST_REGULAR_FILENAME = "./test_regular.csv"

stop_words = set(stopwords.words('english'))
non_alphabet_pattern = re.compile("[^a-zA-Z]")

off_train_dataframe = pandas.read_csv(TRAINING_OFFENSIVE_FILENAME)
off_test_dataframe = pandas.read_csv(TEST_OFFENSIVE_FILENAME)
reg_train_dataframe = pandas.read_csv(TRAINING_REGULAR_FILENAME)
reg_test_dataframe = pandas.read_csv(TEST_REGULAR_FILENAME)


def get_cleaned_tokens(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = [re.sub(non_alphabet_pattern, "", w) for w in filtered_sentence]
    filtered_sentence = [w for w in filtered_sentence if w != ""]
    return filtered_sentence


all_train_tweets = off_train_dataframe['tweet']
all_train_tweets = all_train_tweets.append(reg_train_dataframe['tweet'])
all_train_tweets = [get_cleaned_tokens(tweet) for tweet in all_train_tweets]
all_train_labels = off_train_dataframe["offensive"]
all_train_labels = all_train_labels.append(reg_train_dataframe["offensive"])

all_test_tweets = off_test_dataframe['tweet']
all_test_tweets = all_test_tweets.append(reg_test_dataframe['tweet'])
all_test_tweets = [get_cleaned_tokens(tweet) for tweet in all_test_tweets]
all_test_labels = off_test_dataframe["offensive"]
all_test_labels = all_test_labels.append(reg_test_dataframe["offensive"])

assert len(all_train_tweets) == len(all_train_labels), f"{len(all_train_tweets)} != {len(all_train_labels)}"
assert len(all_test_tweets) == len(all_test_labels), f"{len(all_test_tweets)} != {len(all_test_labels)}"

# tokenize words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(100000)
tokenizer.fit_on_texts(all_train_tweets)
sequences_train = tokenizer.texts_to_sequences(all_train_tweets)
sequences_test = tokenizer.texts_to_sequences(all_test_tweets)

print('Found %s unique tokens.' % len(tokenizer.word_index))

# pad sequences
train_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

embeddings_index = {}
import os

f = open(os.path.join(PRECOMPILED_WV_FILENAME))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

from numpy import array
from numpy import asarray
from numpy import zeros

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((VOCAB_SIZE, DIMENSION_SIZE))  # DIMENSIONS!
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

"""
MAIN MODEL
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D

model = Sequential()
e = Embedding(VOCAB_SIZE, DIMENSION_SIZE, weights=[embedding_matrix],
              input_length=MAX_SEQUENCE_LENGTH, trainable=False)
model.add(e)
for _ in range(LAYER_COUNT):
    model.add(Conv1D(FILTER_NUM, KERNEL_SIZE, activation='relu'))
    if USE_MAX_POOL:
        model.add(GlobalMaxPooling1D())
model.add(Conv1D(FILTER_NUM, KERNEL_SIZE, activation='relu'))
if USE_MAX_POOL:
    model.add(GlobalMaxPooling1D())
model.add(Flatten())  # must use Flatter or GlobalMax
model.add(Dense(DENSE_SIZE, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(DENSE_SIZE, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(train_data, all_train_labels, epochs=5, verbose=0)
# evaluate the model

loss, accuracy = model.evaluate(test_data, all_test_labels, verbose=0)
print("Loss: {}".format(loss))
print('Accuracy: %f' % (accuracy * 100))
training_data_loss, training_data_accuracy= model.evaluate(train_data, all_train_labels, verbose=0)
print("Loss: {}".format(training_data_loss))
print('Accuracy: %f' % (training_data_accuracy* 100))


def vectorize_sentence(tokenizer, sentence):
    cleaned_words = get_cleaned_tokens(sentence)
    tokenized_sentence = tokenizer.texts_to_sequences([cleaned_words])
    return pad_sequences(tokenized_sentence, maxlen=MAX_SEQUENCE_LENGTH)


test_sentences = [
    """Michael Jackson is a miserable human being.""",
]
for i, test_sentence in enumerate(test_sentences):
    print(f"Custom test {i}")
    print(test_sentence)
    vectorized = vectorize_sentence(tokenizer, test_sentence)
    print(model.predict_classes(vectorized))

import uuid
import datetime
current_time = datetime.datetime.now().isoformat()
filename = f"./results/{LAYER_COUNT}-{DIMENSION_SIZE}-{FILTER_NUM}-{KERNEL_SIZE}-{DENSE_SIZE}"
if USE_MAX_POOL:
    filename += "-maxpool"
with open(filename, "a") as f:
    f.write(f"{current_time}\n")
    f.write(f"Dimension Size: {DIMENSION_SIZE}\n")
    f.write(f"Filter Size: {FILTER_NUM}\n")
    f.write(f"Kernel Size: {KERNEL_SIZE}\n")
    f.write("Loss: {}\n".format(loss))
    f.write('Accuracy: %f\n' % (accuracy * 100))
    f.write("\n")