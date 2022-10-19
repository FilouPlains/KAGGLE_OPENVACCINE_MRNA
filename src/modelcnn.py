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
from keras.layers import Reshape
from keras.layers import Multiply
import tensorflow as tf
import numpy as np
import masking

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


mask = masking.masked(2400,130,3,68)
inputs = masking.masked(2400,130,3,68)


def modelcnn(inputs,mask):
    inputs = Input(shape=(130,3))
    inputs2 = Input(shape=(130,3))
    x = Conv1D(32, 3, activation='relu', kernel_initializer='he_uniform',
               input_shape=(130, 1),padding="same")(inputs)
    x = Conv1D(64, 3, activation='relu', kernel_initializer='he_uniform',
               padding="same")(x)
    x = Conv1D(64, 3, activation='relu', kernel_initializer='he_uniform',
               padding="same")(x)
    x = Conv1D(3, 3, activation='relu', kernel_initializer='he_uniform',
               padding="same")(x)
    x = Multiply()([x, inputs2])
    output = Dense(5, activation='linear')(x)

    model = Model(inputs=[inputs,inputs2], outputs=output)
    model.compile(optimizer="adam", loss=MCRMSE)
    return model

model = modelcnn(inputs,mask)
print(model.summary())

# plot_model(model, show_shapes=True, show_layer_names=True)