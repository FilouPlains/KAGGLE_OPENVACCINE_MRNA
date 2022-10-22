"""Main program to launch the neural network learning.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


import numpy as np

import input_network.own_embedding as oweb
import input_network.embedding as emb
import input_network.masking as mask
import input_network.keras_embedding as kreb

import parsing as prs

import neural_network.cnn as cnn

from keras.layers import Add
from keras import Input


if __name__ == "__main__":
    arguments = parsing()

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
    # Creating the input embedding matrix.
    emb_seq: np.array = oweb.input_embedding(data_train[:, 1], emb.BASE)
    emb_sec: np.array = oweb.input_embedding(data_train[:, 2], emb.PAIRED)
    emb_loop: np.array = oweb.input_embedding(data_train[:, 3], emb.LOOP)

    mask = mask.mask(2400, 130, 5, 68)

    keras_input: np.array = kreb.concat_data(data_train)

    # =======================
    #
    # CREATING NEURAL NETWORK
    #
    # =======================
    # OWN_EMBEDDING.
    input_seq, orig_seq = oweb.normalize_input_shape((130, 4), 120)
    input_sec, orig_sec = oweb.normalize_input_shape((130, 3), 120)
    input_loop, orig_loop = oweb.normalize_input_shape((130, 7), 120)

    inputs = Add()([input_seq, input_sec, input_loop])

    model = cnn.cnn(inputs, [orig_seq, orig_sec, orig_loop], Input((130, 5)))
    print(model.summary())

    # KERAS EMBEDDING.
    inputs_2, original_2 = kreb.keras_embedding(120)

    model = cnn.cnn(inputs_2, [original_2], Input((130, 5)))
    print(model.summary())

    # RNABERT EMBEDDING.
    inputs_3 = Input(shape=(130, 120))
    original_3 = inputs_3

    model = cnn.cnn(inputs_3, [original_3], Input((130, 5)))
    print(model.summary())
