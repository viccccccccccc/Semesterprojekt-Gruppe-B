import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
import h5py
import random

### Hyperparameters ###
batch_size = 64
anteil_test = 0.2
output_size = 256 * 256
num_epochs = 5
output_every_k = 1
#######################

class MyDataset(tf.data.Dataset):
    def __init__(self, min_idx, excluded_max, reader):
        self.reader = reader
        self.min = min_idx
        self.max = excluded_max
        self.element_spec = 

    def __len__(self):
        return self.max - self.min

    def __getitem__(self, idx):
        if self.min + idx >= self.max:
            raise Exception("Out of Bounds")
        datapoint = self.reader[self.min + idx]
        x = tf.constant(datapoint["X"][:7], dtype=tf.float32)
        y = tf.constant(datapoint["Y"][:][:], dtype=tf.float32)
        return x, y




class H5Reader:
    def __init__(self, hdf5_path):
        self.file = h5py.File(hdf5_path, "r")
        self.key_list = list(self.file.keys())
        random.shuffle(self.key_list)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        if idx >= len(self.key_list):
            raise Exception("Out of Bounds")
        return self.file[self.key_list[idx]]

def train_test_split(anteil_test, hdf5_path):
    reader = H5Reader(hdf5_path)
    split_index = int(len(reader) * (1 - anteil_test))
    return MyDataset(0, split_index, reader), MyDataset(split_index, len(reader), reader)

class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(16384, activation='relu'),
            layers.Dense(output_size)
        ])

    def call(self, x):
        return self.layers(x)

def train(train_dataset, test_dataset):
    train_losses = []
    test_losses = []
    best_model_loss = 1e10
    model = MLP()
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = Adam(learning_rate=0.001)

    for epoch in range(num_epochs):
        loss_sum = 0
        for inputs, labels in train_dataset:
            labels = tf.reshape(labels, (labels.shape[0], -1))
            with tf.GradientTape() as tape:
                outputs = model(inputs)
                loss = criterion(labels, outputs)
                loss_sum += loss.numpy()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (epoch + 1) % output_every_k == 0:
            test_loss = 0.0
            model.compile(optimizer=optimizer, loss=criterion)
            for inputs, labels in test_dataset:
                labels = tf.reshape(labels, (labels.shape[0], -1))
                outputs = model(inputs)
                loss_for_print = criterion(labels, outputs)
                test_loss += loss_for_print.numpy()

            avg_train_loss = loss_sum / len(train_dataset)
            avg_test_loss = test_loss / len(test_dataset)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            if avg_test_loss < best_model_loss:
                best_model_loss = avg_test_loss
                model.save(f'models/model_save_{num_epochs}.h5')

            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {format(avg_train_loss, ".8f")},'
                f' Test Loss: {format(avg_test_loss, ".8f")}, Learning Rate: {format(optimizer.learning_rate.numpy(), ".8f")}'
            )

    np.savez(f'losses/losses_{num_epochs}.npz', name1=train_losses, name2=test_losses)
    return train_losses, test_losses

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Load and split the data
train_dataset, test_dataset = train_test_split(1./3, "data.h5")
train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=len(train_dataset))
test_dataset = test_dataset.batch(batch_size)

print("Dataset loaded and split!")
print("Start training")
train_losses, test_losses = train(train_dataset, test_dataset)
print("Training finished")
