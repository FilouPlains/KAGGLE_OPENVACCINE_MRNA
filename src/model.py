from turtle import shape
import embedding as emb
import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import GRU
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import SpatialDropout1D
from keras.layers import LayerNormalization
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.layers import Reshape
import tensorflow as tf

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)






# DATA IMPORTATION
#
# ================
data_train: np.array = np.load("data/training.npy", allow_pickle=True)

# ======================================
#
# CREATING PRE-EMBEDDING MATRIX TOKENISE
#
# ======================================
# Creating tokenise embedding matrix.

sequence: np.array = emb.hot_encoding(
    dataset=data_train[:, 1], encoder=emb.BASE)
second_strct: np.array = emb.hot_encoding(dataset=data_train[:, 2],
                                          encoder=emb.PAIRED)
loop_type: np.array = emb.hot_encoding(
    dataset=data_train[:, 3], encoder=emb.LOOP)

arr = np.transpose(np.array([sequence, second_strct, loop_type]), axes=[
                   1, 2, 0])  # shape =(2400,107,3)

cols = [7, 9, 11, 13, 15]
predire = data_train[:, cols].tolist()

# print(len(predire[0]))
# taille_mask = len(predire)
# mask = np.array([1]*107 + [0]*23)

def my_model(pred_len = 68):
    inputs = Input(shape=(107,3))
    embed = Embedding(2400,200,input_length=130)(inputs)
    reshaped = Reshape((107, 600), input_shape = (107, 3, 200))(embed)
    hidden = SpatialDropout1D(0.2)(reshaped)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(hidden)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)

    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = Bidirectional(LSTM(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    # x = Conv1D(32, 3, activation='linear', kernel_initializer='he_uniform', input_shape=(15, 22))(embed)
    # x = Conv1D(32, 3, activation='linear', kernel_initializer='he_uniform', input_shape=(15, 22))(x)
    # x = Conv1D(68, 3, activation='linear', kernel_initializer='he_uniform', input_shape=(15, 22))(x)
    x = x[:, :pred_len]
    
    output = Dense(5, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss=MCRMSE)
    return model

model = my_model()


# plot_model(model, show_shapes=True, show_layer_names=True)
print(model.summary())

# y = np.array(predire).reshape(2400,5,68)
# print(y[0])
# print("stop voici y 0 0")

# print(y[0][0])
# print("stop")
# print(y[1])
# print("stop")
# print(y[0][2])

history_inception = model.fit(arr, y = np.array(predire).reshape(2400,68,5), epochs=10, batch_size=100)



print(history_inception.history.keys())
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.plot(history_inception.history["loss"])
plt.show()
