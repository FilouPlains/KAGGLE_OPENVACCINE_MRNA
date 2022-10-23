"""Main program to launch the neural network learning.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

import input_network.own_embedding as oweb
import input_network.embedding as emb
import input_network.masking as mask
import input_network.keras_embedding as kreb
import input_network.cross_validation as cv


from keras.layers import Add
from keras import Input

from parsing import parsing


if __name__ == "__main__":
    arg = parsing()
    file_out = arg["output"]

    if arg["predict_data"] is None:
        dataset: np.array = np.load(arg["input"], allow_pickle=True)
    else:
        model = arg["input"]
        dataset: np.array = np.load(arg["predict_data"], allow_pickle=True)

    if arg["keras_embedding"]:
        inputs, original = kreb.keras_embedding(120)
    elif arg["own_embedding"]:
        input_seq, orig_seq = oweb.normalize_input_shape((130, 4), 120)
        input_sec, orig_sec = oweb.normalize_input_shape((130, 3), 120)
        input_loop, orig_loop = oweb.normalize_input_shape((130, 7), 120)

        inputs = Add()([input_seq, input_sec, input_loop])
        original = [orig_seq, orig_sec, orig_loop]
    elif arg["rnabert_embedding"]:
        inputs = Input(shape=(130, 120))
        original = [inputs]

    if arg["predict_data"] is None:
        cv.cross_val(arg["cnn"], inputs, original, data_input, mask,
                     data_output)
    else:
        

