"""Create an keras input embedding..
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

# [E]
import embedding as emb
# [N]
import numpy as np

# [K]
from keras import Input
from keras.layers import Embedding, Conv2D


def keras_embedding(filtering=1):
    """Return a keras embedding.

    Parameters
    ----------
    filtering : int, optional
        Filter of the convolutional layer, by default 1.

    Returns
    -------
    keras.Input
        A keras input embedding.
    """
    # Set the input shape.
    inputs = Input(shape=(107, 3))
    # Embedding of the data.
    inputs = Embedding(8, 3, input_length=130)(inputs)
    # Convolution layer to have a 1 at the end.
    inputs = Conv2D(filters=filtering, kernel_size=(1, 1),
                    padding="valid")(inputs)

    return inputs


def concat_data(data: np.array) -> np.array:
    """Concatenate data to have a (n, max(130), 3) shape.

    Parameters
    ----------
    data : np.array
        Whole dataset.

    Returns
    -------
    np.array
        Reshape and concatenate data.
    """
    # Extracting data.
    sequence: np.array = emb.hot_encoding(dataset=data[:, 1],
                                          encoder=emb.BASE)
    second_strct: np.array = emb.hot_encoding(dataset=data[:, 2],
                                              encoder=emb.PAIRED)
    loop_type: np.array = emb.hot_encoding(dataset=data[:, 3],
                                           encoder=emb.LOOP)

    # shape = (LINE, 107, 3)
    dataset = np.transpose(np.array([sequence, second_strct, loop_type]),
                           axes=[1, 2, 0])

    return dataset


if __name__ == "__main__":
    # All main variables start with "m_"
    data_train: np.array = np.load("../../data/training.npy",
                                   allow_pickle=True)

    m_data = concat_data(data_train)
    print(keras_embedding(m_data))
