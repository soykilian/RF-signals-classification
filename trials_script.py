import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D,Convolution2D, Bidirectional, LSTM, GRU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn, json
import scipy.io as io
from typing import Any, Dict

Array = np.ndarray
dataset_path = '/mnt/Data/gmr/Dataset/'
path = '/mnt/Data/gmr/models/'
with h5py.File(dataset_path + 'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
print(X_train.shape)
with h5py.File(dataset_path + 'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
print("Signals data loaded")
print("---------------------------------------------------")
lbl_train = io.loadmat(dataset_path + 'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path + 'lbl_test.mat')['lbl_test']

print(lbl_test.shape)
print(lbl_train.shape)
Y_train = io.loadmat(dataset_path + 'Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = io.loadmat(dataset_path + 'Y_test.mat')
Y_test = Y_test['Y_test']
classes = ['LFM', '2FSK', '4FSK', '8FSK', 'Costas', '2PSK', '4PSK', '8PSK', 'Barker', 'Huffman', 'Frank', 'P1', 'P2',
           'P3', 'P4', 'Px', 'Zadoff-Chu', 'T1', 'T2', 'T3', 'T4', 'NM', 'ruido']
I_x = X_train[:, :, 0]
Q_x = X_train[:, :, 1]
X_train[:, :, 1] = np.arctan(Q_x, I_x) / np.pi
X_train[:, :, 0] = np.abs(I_x + 1j * Q_x)
I_t = X_test[:, :, 0]
Q_t = X_test[:, :, 1]
X_test[:, :, 0] = np.arctan(Q_t, I_t) / np.pi
X_test[:, :, 1] = np.abs(I_t + 1j * Q_t)

np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:], Y_train[:], lbl_train[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)
print("Data shuffled")
print("---------------------------------------------------")


def lstm_network(input_shape: int) -> Model:
    X_input = Input(input_shape)
    X = LSTM(128, return_sequences=True, name='lstm0', dropout=0.3)(X_input)
    X = LSTM(128, name='lstm1')(X)
    X = Dense(23, activation='softmax', name='fc0')(X)
    model = Model(inputs=X_input, outputs=X)
    model.summary()
    return model

model = lstm_network(X_train.shape[1:])
print("Model initiated.")
print("---------------------------------------------------")
c = [ModelCheckpoint(filepath='/mnt/Data/gmr/pruebas/bestmodel.h5', monitor='cal_loss', save_best_only=True)]
model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
print("Optimizer added.")
print("---------------------------------------------------")

def Train(model: Model, data: Dict, n_epochs: int = 500, batch_size: int = 150):
    history = model.fit(data['X_train'], data['Y_train'], epochs=n_epochs, batch_size=batch_size, callbacks=c,
                        validation_data=(data['X_test'], data['Y_test']))
    with open(path + 'history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(path + 'model_rnn.json', 'w') as json_file:
        json_file.write(model_json)

def save_best():
    model.load_weights(path + 'best_model.h5')
    with open(output_path + 'history_rnn.json', 'r') as f:
        history = json.load(f)

Train(model, {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test})
