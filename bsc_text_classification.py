#Classify (binary classification)the movie reviews as
#positive or negative using the text of the review

#As dataset, the IMDB dataset containing 50.000 movie
#reviews will be used. (25.000-test, 25.000-train)

#Balanced dataset: a dataset containing equal number
#of positive and negative reviews.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np


#Download the IMDB dataset
#Each integer in the dataset represents a specific word


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


#print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#To see what the first review looks like, uncomment:
#print(train_data[0])



# Create a dictionary to convert integers back to text
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


#To see how decoding looks like, uncomment the line below:
#print(decode_review(train_data[0]))

#PREPARE THE DATA

#The reviews might not be of same size
#Inputs to a neural network must be the same length
#Use pad_sequences function to standardize the lengths

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)



#Build the model

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

#Create the model
model = keras.Sequential()

#Add an embedding layer. (https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12)
#Take an integer encoded vocabulary and look  up the embedding vector for each word-index.
model.add(keras.layers.Embedding(vocab_size, 16))
#Produce a fixed-length output vector for each example by averaging over the sequence dimension.
model.add(keras.layers.GlobalAveragePooling1D())
#Pipe through the fixed length output vector through a fully-connected layer with 16 units
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#Single output node containing a value btw. 0-1 representing the confidence level
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


#binary-crossentropy is selected as the loss function since this suits this
#binary classification problem
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

#Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


#Train the model

tr = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#Evaluate the model

results = model.evaluate(test_data, test_labels)

print(results)

#HOW TO MONITOR THE MODEL TRAINING ACCURACY?

#model.fit function returns a history object that contains a dictionary
#with everything that happened during training

history_dict = tr.history
history_dict.keys() #Key values of the dictionary are: loss, accuracy, validation loss, and validation accuracy

#Let's visualize the loss and accuracy through training process

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


#The validation accuracy shows an example of an overfitting situation.
plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()






#uncomment the line below to see the model summary that has
#just been created
#model.summary()
