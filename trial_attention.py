import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Add, Conv1D,Convolution2D, Bidirectional, LSTM, GRU, AlphaDropout, MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow_addons.layers import MultiHeadAttention
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn, json
import scipy.io as io
from typing import Any, Dict
from includes.clr_callback import *
path = '/home/maria/'
dataset_path = path + 'dataset_1d/'
with h5py.File(dataset_path +'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
print(X_train.shape)
with h5py.File(dataset_path +'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
    lbl_train = io.loadmat(dataset_path +'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path +'lbl_test.mat')['lbl_test']
print(lbl_test.shape)
print(lbl_train.shape)
Y_train = io.loadmat(dataset_path +'Y_train.mat')
Y_train = Y_train['Y_train']
print(Y_train.shape)
Y_test = io.loadmat(dataset_path +'Y_test.mat')
Y_test = Y_test['Y_test']
print(Y_test.shape)
classes = ['LFM', '2FSK', '4FSK', '8FSK', 'Costas', '2PSK', '4PSK', '8PSK', 'Barker', 'Huffman', 'Frank', 'P1', 'P2',
           'P3', 'P4', 'Px', 'Zadoff-Chu', 'T1', 'T2', 'T3', 'T4', 'NM', 'ruido']
I_x = X_train[:, :, 0]
Q_x = X_train[:, :, 1]
X_train[:,:,1] = np.arctan(Q_x, I_x)/ np.pi
X_train[:,:,0] = np.abs(I_x + 1j*Q_x)
I_t = X_test[:, :, 0]
Q_t = X_test[:, :, 1]
X_test[:, :, 0] = np.arctan(Q_t, I_t)/np.pi
X_test[:, :, 1] = np.abs(I_t + 1j*Q_t)
np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:],Y_train[:], lbl_train[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)

def ModelTrunk(input_shape : int):
    X_input = tf.keras.Input(input_shape)
    num_layers = 5
    #attention_layers = [AttentionBlock(num_heads=2, head_size=128, ff_dim=None, dropout=0.1) for _ in range (num_layers)]
    #for layer in attention_layers :
      #  x = layer(x)
    attention_block = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
    x = attention_block(X_input,X_input)
    x = Flatten()(x)
    x = Dense(128, activation='selu')(x)
    x = AlphaDropout(0.6)(x)
    x = Dense(128, activation='selu')(x)
    x = AlphaDropout(0.6)(x)
    x = Dense(23, activation='softmax')(x)
    model = Model(inputs = X_input, outputs = x)
    model.summary()
    return model
model = ModelTrunk(X_train.shape[1:])
model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])
output_path = path + 'Results/model_stft'
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-4, step_size= 4 * (X_train.shape[0] // 256))
c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(X_train, Y_train, epochs = 500, batch_size = 256, callbacks = c, validation_data=(X_test, Y_test))
with open(output_path +'history_rnn.json', 'w') as f:
    json.dump(history.history, f)
model_json = model.to_json()
with open(output_path +'model_rnn.json', "w") as json_file:
    json_file.write(model_json)
