"""Create en embedding matrix with given `string` data.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

import numpy as np

BASE = {
    "A": 0,
    "U": 1,
    "G": 2,
    "C": 3
}

PAIRED = {
    "(": 0,
    ".": 1,
    ")": 2
}

LOOP = {
    "E": 0,
    "B": 1,
    "M": 2,
    "X": 3,
    "I": 4,
    "H": 5,
    "S": 6
}


def hot_encoding(array: np.array(str), dic: dict[str, int]) \
        -> np.array(np.array(int)):
    """Modify a numpy array of string letters into a one with string digits.

    Parameters
    ----------
    array : np.array(str)
        The array to modify.
    dic : dict[str, str]
        Dictionary containing which characters to modify (`key`) with what
        (`value`).

    Returns
    -------
    np.array(np.array(int))
        The modify array.
    """
    # Break string into list, with 1 character per index.
    brk_array: list[list[str]] = [*array]
    int_array: list[list[int]] = []

    # Parsing all array's row.
    for row in brk_array:
        trlt_base: list[int] = []

        # Parsing all row's base.
        for base in row:
            trlt_base.append(dic[base])

        int_array.append(trlt_base)

    # Returning into a numpy array.
    return np.array(int_array)


def positional_embedding(length: int, seq_len: int) \
        -> np.array(np.array(float)):
    """Return positional embedding of a given length (`row`) and seq_len
    (`column`).

    Parameters
    ----------
    length : int
        Number of row in the embedding
    seq_len : int
        The length of the sequence.

    Returns
    -------
    np.array(np.array(float))
        The positional embedding for given data.
    """
    # Create an empty array.
    array: np.array = np.zeros([length, seq_len])
    # Create a array of `seq_len` size.
    to_filter: np.array = np.array(range(seq_len))
    # Create a array of `length` size.
    dimension: np.array = np.transpose(np.array([range(length)] * seq_len))

    # To filter odd and even index.
    o_f: np.array = to_filter % 2 == 0
    e_f: np.array = to_filter % 2 == 1

    # Doing arithmetic operation applied with a vectoriel
    array[:, o_f] = np.sin(np.divide(dimension[:, o_f], 10_000 **
                                     (2 * to_filter[o_f] / seq_len)))
    array[:, e_f] = np.cos(np.divide(dimension[:, e_f], 10_000 **
                                     (2 * to_filter[e_f] / seq_len)))

    return array


if __name__ == '__main__':
    # Data importation.
    data_train: np.array = np.load("../data/training.npy", allow_pickle=True)

    # Creating input embedding matrix.
    sequence: np.array = hot_encoding(array=data_train[:, 1], dic=BASE)
    second_strct: np.array = hot_encoding(array=data_train[:, 2], dic=PAIRED)
    loop_type: np.array = hot_encoding(array=data_train[:, 3], dic=LOOP)

    # Creating positional embedding matrix.
    dim: tuple = sequence.shape
    pos: np.array = positional_embedding(length=dim[0], seq_len=dim[1])

    # Creating the embedding matrix.
    emb_seq: np.array = sequence + pos
    emb_sec: np.array = second_strct + pos
    emb_loop: np.array = loop_type + pos
