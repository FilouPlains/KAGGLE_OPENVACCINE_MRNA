"""Masked vector for the ouput of the model
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

import numpy as np


def mask(line: int, col: int, depth: int, desire_output: int) -> np.array:
    """Return a mask to apply to a neural network output.

    Parameters
    ----------
    line : int
        Number of lines.
    col : int
        Number of columns.
    depth : int
        3D size/depth. Size of the data array.
    desire_output : int
        Which column from begin to end to set to `1`.

    Returns
    -------
    np.array
        The 3D matrix mask.
    """
    masking: np.array = np.zeros((line, col, depth))
    masking[:, 0:desire_output, ] = 1

    return masking


def mask_test(data_test, line: int, col: int, depth: int) -> np.array:
    """Return a mask to apply to a neural network output for our test.

    Parameters
    ----------
    data_test : numpy array
        Input of our network
    line : int
        Number of lines.
    col : int
        Number of columns.
    depth : int
        3D size/depth. Size of the data array.

    Returns
    -------
    np.array
        The 3D matrix mask.
    """
    masking: np.array = np.zeros((line, col, depth))

    for i in range(line):
        if len(data_test[i]) == 107:
            masking[:, 0:68, ] = 1
        if len(data_test[i]) == 130:
            masking[:, 0:91, ] = 1

    return masking


def format_input(input_seq, line: int, depth: int):
    """Return a input to apply to a neural network input for our train.

    Parameters
    ----------
    input_seq : numpy array
        Input with sequence of 107 of our network
    line : int
        Number of lines.
    depth : int
        3D size/depth. Size of the data array.

    Returns
    -------
    np.array
        Input with sequence of 130 of our network
    """
    seq_input_yes: np.array = np.zeros((line, 23, depth))

    seq_input_yes: np.array = np.concatenate((input_seq, seq_input_yes),
                                             axis=1)

    return seq_input_yes


if __name__ == "__main__":
    # Load data to see the lengt of the sequences
    data_test: np.array = np.load("../../data/bert_test.npy",
                                  allow_pickle=True)

    print(mask_test(data_test, 3634, 130, 5))

    data_test: np.array = np.load("../../data/bert_train.npy",
                                  allow_pickle=True)

    print(format_input(data_test, 2400, 120))
