"""Google inception neural network.
"""

# [N]
from neural_network.mcrmse import mcrmse
# [K]
from keras import Input, Model
from keras.layers import Conv1D, Dense, concatenate, MaxPooling1D, Multiply
from keras.layers import Bidirectional, GRU


__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


def __inception(inputs):
    """Return a block of inception.

    Parameters
    ----------
    inputs : keras.Input
        The input data.

    Returns
    -------
    keras.layers
        a block of inception.
    """
    # First convolution.
    conv_1 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv_1 = Conv1D(22, (3), padding="same", activation="relu")(conv_1)
    # Second convolution.
    conv_2 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv_2 = Conv1D(22, (5), padding="same", activation="relu")(conv_2)
    # Third convolution.
    conv_3 = MaxPooling1D((3), strides=(1), padding="same")(inputs)
    conv_3 = Conv1D(22, (1), padding="same", activation="relu")(conv_3)
    # Last convolution.
    conv_4 = Conv1D(22, (1), padding="same", activation="relu")(inputs)
    conv = concatenate([conv_1, conv_2, conv_3, conv_4], axis=2)

    return conv


def inception(inputs, original, mask, n_inception=2):
    """Apply a inception to a given input, with a given mask.

    Parameters
    ----------
    inputs : keras.Input
        The input data.
    original : keras.Input
        The original input data.
    mask : mask
        The mask.
    n_inceptions : int
        Number of inceptions block.
    Returns
    -------
    Model.compile
        A compile model to be used for training.
    """
    print("=" * 80 + "\n")
    print(f"Doing a google inception neural network with {n_inception} "
          "modules.\n")
    print("=" * 80 + "\n")

    # Inception layers.
    for _ in range(n_inception):
        inputs = __inception(inputs)

    inputs = Dense(1200, activation="relu")(inputs)

    # Dense layers.
    inputs = Dense(600, activation="relu")(inputs)
    inputs = Dense(150, activation="relu")(inputs)
    inputs = Dense(5, activation="relu")(inputs)

    inputs = Multiply()([inputs, mask])
    output = Dense(5, activation="linear")(inputs)

    # Set the model.
    model = Model(inputs=original + [mask], outputs=output)

    # Compile then return the model.
    model.compile(optimizer="adam", loss=mcrmse)

    return model


if __name__ == "__main__":
    inputs_3 = Input(shape=(130, 120))
    original_3 = inputs_3

    inception(inputs_3, [original_3], Input((130, 5)))
