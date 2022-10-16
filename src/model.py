from importlib_metadata import _top_level_declared
from matplotlib.cbook import to_filehandle
from matplotlib.pyplot import axes
import embedding as emb
import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout1D
from keras.layers import Activation
from keras.layers import MaxPooling1D
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.layers import Reshape
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

arr = np.transpose(np.array([sequence, second_strct, loop_type]), axes=[ 1, 2, 0])  # shape =(2400,107,3)
print(arr.shape)


cols = [7, 9, 11, 13, 15]
predire = np.transpose(data_train[:, cols]).tolist()
print(len(predire[0]))



def my_model(pred_len = 68):
    inputs = Input(shape=(107,3))
    embed = Embedding(2400,200,input_length=130)(inputs)
    reshaped = Reshape((107, 600), input_shape = (107, 3, 200))(embed)
    x = Conv1D(512, 3,padding='same',kernel_initializer='he_uniform') (reshaped)
    x = BatchNormalization() (x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal')(x)
    x = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(x)
    x = LSTM(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal')(x)
    x = x[:, :pred_len]
    output = x #(None, 68, 5)

    output = Dense(5, activation='linear')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = my_model()


# plot_model(model, show_shapes=True, show_layer_names=True)
print(model.summary())

history_inception = model.fit(arr, y = np.array(predire).reshape(2400,68,5), epochs=5, batch_size=100)
model.evaluate(arr, y = np.array(predire).reshape(2400,68,5))

# print(history_inception.history.keys())
# plt.plot(history_inception.history['accuracy'])
# plt.plot(history_inception.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# plt.plot(history_inception.history['loss'])
# plt.plot(history_inception.history['val_loss'])
# plt.show()
