
#Predict the output of a continuos variable such as price or probability
#in contrast to a classification problem, where we are trying to predict
# the class that the sample belongs to.

#For this, the UCI Auto MPG dataset will be used.

#Required libraries: seaborn, pandas, matplotlib, tensorflow


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


#Download the Auto MPG dataset

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


#Import the dataset using pandas

#Determine the column names

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

#The last elements of the dataset
tail = dataset.tail()

#The rows with unknown values
na = dataset.isna().sum()


dataset = dataset.dropna()


#The origin colums has a categorical representation. First,
#convert it to one-hot encoding:
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0


#Divide the dataset into a training and a test set, if you want the
#rows to be selected randomly, change the random_state variable
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#Inspect the data- very useful function!
#Have a quick look at the joint distribution of a few pairs of columns from the training set.
#Don't forget to show the plot

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()


#Look at the overall statistics-
#Generate descriptive statistics that summarize the central tendency,
#dispersion and shape of a dataset’s distribution.
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

#Separate the labels from the features.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')



#Define a normalization function and normalize the training and testing data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#Build the model using a Sequential model with 2 densely connected hidden layers
#and an output layer that returns a single, sontinuous value.

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model= build_model()

#In order to print the description of the model, use the .summary function
model.summary()


#Try the model by taking a batch of 10 samples and call model.predict on it
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

#Train the model for 1000 epochs, and record the training and validation accuracy in the history object.

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)


#Evaluate the model on the test set
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(mae))


#Make predictions on the test set for MPG values

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()


#How to look at the error distribution

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()


