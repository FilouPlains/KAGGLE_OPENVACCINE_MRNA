"""Create en embedding matrix with given `string` data.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# E
import embedding as emb
# N
import numpy as np

# K
from keras.layers import Activation, Add, BatchNormalization, Conv2D
from keras import Input


def normalize_input_shape(shape: "tuple[int]", filtering: int = 1):
    """Take a shape and return a normalized `keras.Input` shape.

    Parameters
    ----------
    shape : tuple[int]
        Data's shape.
    filtering : int, optional
        Filtering of convolutional layer, by default 1.

    Returns
    -------
    keras.Input
        The `keras.Input` with a normalize shape.
    """
    inputs = Input(shape=shape)
    original = inputs
    inputs = BatchNormalization()(inputs)
    inputs = Activation("relu")(inputs)
    inputs = Conv2D(filters=filtering, kernel_size=(1, 1),
                    padding="valid")(inputs)
    inputs = Add()([inputs, original])
    inputs = Conv2D(filters=filtering, kernel_size=(1, 1),
                    padding="valid")(inputs)

    return inputs


def input_embedding(data: np.array, encoder: "dict[str, int]") -> np.array:
    """Take data and return an input embedding.

    Parameters
    ----------
    data : np.array
        The data to transform.
    encoder : dict[str, int]
        How to hot encode the embedding.

    Returns
    -------
    np.array
        The input embedding.
    """
    # Creating tokenise embedding matrix.
    hot_embedding: np.array = emb.hot_encoding(
        dataset=data,
        encoder=encoder,
        is_tokenise=False
    )

    # Creating positional embedding matrix.
    dim_seq: tuple = hot_embedding.shape
    pos_embedding: np.array = emb.positional_embedding(
        length=dim_seq[0],
        seq_len=dim_seq[1],
        is_tokenise=False,
        len_tok_arr=dim_seq[2]
    )

    return hot_embedding + pos_embedding


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
    # Creating the input embedding matrix.
    emb_seq: np.array = input_embedding(data_train[:, 1], emb.BASE)
    emb_sec: np.array = input_embedding(data_train[:, 2], emb.PAIRED)
    emb_loop: np.array = input_embedding(data_train[:, 3], emb.LOOP)

    # =======================
    #
    # CREATING NEURAL NETWORK
    #
    # =======================
    input_seq = normalize_input_shape((107, 4))
    input_sec = normalize_input_shape((107, 3))
    input_loop = normalize_input_shape((107, 7))

    m_inputs = Add()([input_seq, input_sec, input_loop])
