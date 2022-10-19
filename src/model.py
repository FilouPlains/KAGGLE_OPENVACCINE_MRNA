from turtle import shape
import embedding as emb
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import Dense
from keras import Model
import modelcnn as md
import masking


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


input_sec = Input(shape=(2400,130, 3))
input_sec = BatchNormalization()(input_sec)
input_sec = Activation("relu")(input_sec)
input_sec = Conv2D(filters=1, kernel_size=(1, 1), padding="valid")(input_sec)
input_sec = Add()([input_sec, arr])
input_sec = Conv2D(filters=1, kernel_size=(1, 1),
                    padding="valid")(input_sec)
output = input_sec

model = Model(input_sec, output)

print(model.summary())

mask = masking.masked(2400,130,1,68)

modelcnn = md.modelcnn(arr,mask)




# y = np.array(predire).reshape(2400,5,68)
# print(y[0])
# print("stop voici y 0 0")

# print(y[0][0])
# print("stop")
# print(y[1])
# print("stop")
# print(y[0][2])

history_modelcnn = modelcnn.fit(arr, y = np.array(predire).reshape(2400,68,5), 
                                epochs=10, batch_size=100)



print(history_modelcnn.history.keys())
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.plot(history_modelcnn.history["loss"])
plt.show()
