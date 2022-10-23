"""Calculate the mcmre of train and validation.
"""
from neural_network.cnn import cnn
import input_network.own_embedding as oweb
import numpy as np
from input_network.masking import mask
import input_network.embedding as emb
from keras.layers import Add
import matplotlib.pyplot as plt
from neural_network.cnn import cnn
import input_network.keras_embedding as kreb
from keras import Input

import input_network.neural_network.cnn as cnn


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
        model = cnn.cnn(inputs, original, Input((130, 5)))

    print(model.summary())

    # Fitting the model
    history = model.fit([data_input, mask], data_output, validation_split=0.2,
                        epochs=20, batch_size=100)

    return model, history


if __name__ == "__main__":
    #  charge data
    data_train: np.array = np.load("data/training.npy", allow_pickle=True)

    # masking vector
    masked = mask(2400, 130, 5, 68)

    # Output for the 3 model
    cols = [7, 9, 11, 13, 15]
    predire = data_train[:, cols].tolist()
    predire = np.array(predire).reshape(2400, 68, 5)
    predire_yes = np.zeros((2400, 62, 5))
    predire = np.concatenate((predire, predire_yes), axis=1)

    # Own embedding
    # Input for the model one
    emb_seq: np.array = oweb.input_embedding(data_train[:, 1], emb.BASE)
    emb_sec: np.array = oweb.input_embedding(data_train[:, 2], emb.PAIRED)
    emb_loop: np.array = oweb.input_embedding(data_train[:, 3], emb.LOOP)

    emb_seq_yes: np.array = np.zeros((2400, 23, 4))
    emb_seq = np.concatenate((emb_seq, emb_seq_yes), axis=1)

    emb_sec_yes: np.array = np.zeros((2400, 23, 3))
    emb_sec = np.concatenate((emb_sec, emb_sec_yes), axis=1)

    emb_loop_yes: np.array = np.zeros((2400, 23, 7))
    emb_loop = np.concatenate((emb_loop, emb_loop_yes), axis=1)

    # Formatting the shape of input for the model 1
    input_seq, orig_seq = oweb.normalize_input_shape((130, 4), 120)
    input_sec, orig_sec = oweb.normalize_input_shape((130, 3), 120)
    input_loop, orig_loop = oweb.normalize_input_shape((130, 7), 120)
    m_inputs = Add()([input_seq, input_sec, input_loop])

    # Creating model 1
    model1 = cnn(m_inputs, [orig_seq, orig_sec, orig_loop], Input((130, 5)))

    # Validation
    cross_val(model1, [emb_seq, emb_sec, emb_loop],
              masked, predire, "own_fig")

    # keras embedding
    # Input for the model 2
    sequence: np.array = emb.hot_encoding(dataset=data_train[:, 1],
                                          encoder=emb.BASE)
    second_strct: np.array = emb.hot_encoding(dataset=data_train[:, 2],
                                              encoder=emb.PAIRED)
    loop_type: np.array = emb.hot_encoding(dataset=data_train[:, 3],
                                           encoder=emb.LOOP)
    data_input = np.transpose(np.array([sequence, second_strct, loop_type]),
                              axes=[1, 2, 0])
    data_input_yes: np.array = np.zeros((2400, 23, 3))
    data_input_yes = np.concatenate((data_input, data_input_yes), axis=1)

    # Formatting the shape of input for the model 2
    inputs_2, original_2 = kreb.keras_embedding(120)

    # Creating model 2
    model2 = cnn(inputs_2, [original_2], Input((130, 5)))

    # Validation
    cross_val(model2, data_input_yes, masked, predire, "keras_fig")

    # RNABERT embedding
    # Formatting the shape of input for the model 3
    inputs_3 = Input(shape=(130, 120))
    original_3 = inputs_3

    # Input for the model 3
    input_seq: np.array = np.load("data/bert_train.npy", allow_pickle=True)
    seq_input_yes: np.array = np.zeros((2400, 23, 120))
    seq_input_yes = np.concatenate((input_seq, seq_input_yes), axis=1)

    # Creating model 3
    model3 = cnn(inputs_3, [original_3], Input((130, 5)))

    # Validation
    cross_val(model3, seq_input_yes, masked, predire, "RNABERT_fig")
