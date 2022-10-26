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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


EPOCHS = 100
BATCH_SIZE = 50


def cross_val(is_nn_cnn, inputs, original, data_input, mask, data_output,
              file_out):
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
    file_out :
        Name of the file for saving the figure and the model.
    """
    if is_nn_cnn:
        model = cnn.cnn(inputs, original, Input((130, 5)))
    else:
        model = inc.inception(inputs, original, Input((130, 5)))

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(file_out , save_best_only=True, monitor='val_loss', mode='min')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, epsilon=1e-4, mode='min')

    # Fitting the model
    history = model.fit([data_input, mask], data_output, validation_split=0.2,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks= [earlyStopping,
                        mcp_save])

    print("\n" + "=" * 80 + "\n")
    print("Writting `data/history.csv` to access to 'LOSS' and 'VAL_LOSS'.\n")
    print("=" * 80 + "\n")

    file_name: str = file_out[:-3] + "_history.csv"

    with open(file_name, "w", encoding="utf8") as file:
        file.write(f"LOSS,VAL_LOSS\n")

        for i, loss in enumerate(history.history["loss"]):
            file.write(f"{loss},{history.history['val_loss'][i]}\n")

        file.close()

    return model, history
