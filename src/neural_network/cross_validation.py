"""Calculate the mcmre of train and validation.
"""
import numpy as np

import input_network.embedding as emb
import input_network.keras_embedding as kreb
import input_network.own_embedding as oweb

from neural_network.cnn import cnn
from input_network.masking import mask
from keras.layers import Add
from neural_network.cnn import cnn
from keras import Input

import neural_network.cnn as cnn
import neural_network.inception as inc


EPOCHS = 1


def cross_val(is_nn_cnn, inputs, original, data_input, mask, data_output):
    """Calculate the mcmre of train and val and save the figure.

    Parameters
    ----------
    model_cnn : model
        a model compile
    data_input : 
        Data input that will be given to the model
    masked : np.array
        A vector of masking
    data_output :
        Data input that will be given to the model
    name_file :
        Name of the file for saving the figure

    """
    if is_nn_cnn:
        model = cnn.cnn(inputs, original, Input((130, 5)))
    else:
        model = inc.inception(inputs, original, Input((130, 5)))

    print(model.summary())

    # Fitting the model
    history = model.fit([data_input, mask], data_output, validation_split=0.2,
                        epochs=EPOCHS, batch_size=100)

    return model, history
