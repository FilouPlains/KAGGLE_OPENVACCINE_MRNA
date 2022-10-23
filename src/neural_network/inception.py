"""Google inception neural network.
"""

# [N]
from neural_network.mcrmse import mcrmse
# [K]
from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, concatenate, MaxPooling1D,Multiply,Add,Bidirectional,GRU,LSTM
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# import input_network.own_embedding as oweb
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

def __inception(inputs):
    # 1st
    conv_1 = Conv1D(22, (1), padding='same', activation='relu')(inputs)
    conv_1 = Conv1D(22, (3), padding='same', activation='relu')(conv_1)
    # 2
    conv_2 = Conv1D(22, (1), padding='same', activation='relu')(inputs)
    conv_2 = Conv1D(22, (5), padding='same', activation='relu')(conv_2)
    # 3
    conv_3 = MaxPooling1D((3), strides=(1), padding='same')(inputs)
    conv_3 = Conv1D(22, (1), padding='same', activation="relu")(conv_3)
    # 4
    conv_4 = Conv1D(22, (1), padding='same', activation='relu')(inputs)
    conc_1 = concatenate([conv_1, conv_2, conv_3, conv_4], axis=2)

    return conc_1


def inception(inputs, original, mask):
    n_inception = 2
    print("Simple inception network with %d modules" % (n_inception))
    # inputs = Input(shape=(130, 1))

    # layer
    inception_i = inputs
    for _ in range(n_inception):
        inception_i = __inception(inception_i)

#   flat_1 = Flatten()(inception_i)
    dense_1 = Dense(1200, activation='relu')(inception_i)  # (flat_1)
    dense_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(dense_1)
    dense_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(dense_1)
    dense_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(dense_1)
    dense_1 = Bidirectional(GRU(256, dropout=0.2, return_sequences=True, kernel_initializer='orthogonal'))(dense_1)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)
    dense_4 = Dense(5, activation='relu')(dense_3)
    
    masked = Multiply()([dense_4, mask])
    output = Dense(5, activation='linear')(masked)


    # Set the model.
    model = Model(inputs=original + [mask], outputs=output)

    # Compile then return the model.
    model.compile(optimizer="adam", loss=__mcrmse)
    print(model.summary())
    return model

if __name__ == "__main__":
    inputs_3 = Input(shape=(130, 120))
    original_3 = inputs_3

    inception(inputs_3, [original_3], Input((130, 5)))