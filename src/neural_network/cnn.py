"""Create a CNN neural network.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


import tensorflow as tf

from keras import Model
from keras.layers import Dense, Conv1D, Multiply


def __mcrmse(y_true, y_pred):
    """[PRIVATE]Loss function

    Parameters
    ----------
    y_true : float
        The actual `y`.
    y_pred : float
        The `y` to predict.

    Returns
    -------
    rmse
        The compute rmse.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(mse), axis=1)


def cnn(inputs, mask):
    """Apply a CNN to a given input, with a given mask.

    Parameters
    ----------
    inputs : keras.Input
        The input data.
    mask : keras.Input
        The mask.

    Returns
    -------
    Model.compile
        A compile model to be used for training.
    """
    # Neural network.
    network = Conv1D(32, 3, activation="relu", kernel_initializer="he_uniform",
                     input_shape=(130, 1), padding="same")(inputs)
    network = Conv1D(64, 3, activation="relu", kernel_initializer="he_uniform",
                     padding="same")(network)
    network = Conv1D(64, 3, activation="relu", kernel_initializer="he_uniform",
                     padding="same")(network)
    network = Conv1D(3, 3, activation="relu", kernel_initializer="he_uniform",
                     padding="same")(network)
    # Applying the mask.
    network = Multiply()([network, mask])

    # Set the output.
    output = Dense(5, activation="linear")(network)

    # Set the model.
    model = Model(inputs=[inputs, mask], outputs=output)

    # Compile then return the model.
    return model.compile(optimizer="adam", loss=__mcrmse)
