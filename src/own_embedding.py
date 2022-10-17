"""Create en embedding matrix with given `string` data.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

import numpy as np
import embedding as emb

from keras.layers import GRU

# ========
#
# IMPORT *
#
# ========
import matplotlib.pyplot as plt
import numpy as np

# ===============
#
# FROM * IMPORT *
#
# ===============
# K
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv3D, Dense
from keras.layers import Flatten
from keras import Input, Model
from keras.layers import Activation
from keras.utils import np_utils
import keras
# O
from os import system
# S
from sklearn.model_selection import train_test_split
from sys import exit
from keras.layers import concatenate


if __name__ == "__main__":
    # ================
    #
    # DATA IMPORTATION
    #
    # ================
    data_train: np.array = np.load("data/training.npy", allow_pickle=True)

    # =============================
    #
    # CREATING PRE-EMBEDDING MATRIX
    #
    # =============================
    # Creating tokenise embedding matrix.
    sequence: np.array = emb.hot_encoding(dataset=data_train[:, 1],
                                          encoder=emb.BASE, is_tokenise=False)
    second_strct: np.array = emb.hot_encoding(
        dataset=data_train[:, 2],
        encoder=emb.PAIRED,
        is_tokenise=False
    )
    loop_type: np.array = emb.hot_encoding(dataset=data_train[:, 3],
                                           encoder=emb.LOOP, is_tokenise=False)

    # Creating positional embedding matrix.
    dim_seq: tuple = sequence.shape
    pos_seq: np.array = emb.positional_embedding(
        length=dim_seq[0],
        seq_len=dim_seq[1],
        is_tokenise=False,
        len_tok_arr=dim_seq[2]
    )
    dim_sec: tuple = second_strct.shape
    pos_sec: np.array = emb.positional_embedding(
        length=dim_sec[0],
        seq_len=dim_sec[1],
        is_tokenise=False,
        len_tok_arr=dim_sec[2]
    )
    dim_loop: tuple = loop_type.shape
    pos_loop: np.array = emb.positional_embedding(
        length=dim_loop[0],
        seq_len=dim_loop[1],
        is_tokenise=False,
        len_tok_arr=dim_loop[2]
    )

    # Creating the input embedding matrix.
    emb_seq: np.array = sequence + pos_seq
    emb_sec: np.array = second_strct + pos_sec
    emb_loop: np.array = loop_type + pos_loop

    # =======================
    #
    # CREATING NEURAL NETWORK
    #
    # =======================

    filtering = 1

    ###########################
    # INPUT ###################
    ###########################

    input_seq = Input(shape=(2400, 107, 4))
    original_seq = input_seq
    input_seq = BatchNormalization()(input_seq)
    input_seq = Activation("relu")(input_seq)
    input_seq = Conv2D(filters=filtering, kernel_size=(1, 1),
                       padding="valid")(input_seq)
    input_seq = Add()([input_seq, original_seq])
    input_seq = Conv2D(filters=filtering, kernel_size=(1, 1),
                       padding="valid")(input_seq)

    input_sec = Input(shape=(2400, 107, 3))
    original_sec = input_sec
    input_sec = BatchNormalization()(input_sec)
    input_sec = Activation("relu")(input_sec)
    input_sec = Conv2D(filters=filtering, kernel_size=(1, 1),
                       padding="valid")(input_sec)
    input_sec = Add()([input_sec, original_sec])
    input_sec = Conv2D(filters=filtering, kernel_size=(1, 1),
                       padding="valid")(input_sec)

    input_loop = Input(shape=(2400, 107, 7))
    original_loop = input_loop
    input_loop = BatchNormalization()(input_loop)
    input_loop = Activation("relu")(input_loop)
    input_loop = Conv2D(filters=filtering, kernel_size=(1, 1),
                        padding="valid")(input_loop)
    input_loop = Add()([input_loop, original_loop])
    input_loop = Conv2D(filters=filtering, kernel_size=(1, 1),
                        padding="valid")(input_loop)

    # input_data = Add()([input_seq, input_sec, input_loop, original_seq,
    #                     original_sec, original_loop])
    # input_sec = Add()([input_data, input_seq, input_sec, input_loop])
    input_data = concatenate([input_seq, input_sec, input_loop], axis = 2)
    original = original_loop

    ###########################
    # INTERN TREATMENT ########
    ###########################

    # Bidirectional(GRU(256, drouput=0.2, return_sequences=True,
    #                   kernel_initializer="orthogonal"))(x)

    ###########################
    # OUTPUT ##################
    ###########################
    # Adding a flatten layer.
    input_data = Flatten()(input_data)
    output = Dense(3, activation="softmax")(input_data)

    # Create the model.
    model = Model(original, output)

    # Compile model.
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())
