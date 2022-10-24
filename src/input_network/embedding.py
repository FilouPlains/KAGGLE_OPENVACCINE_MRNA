"""Create an embedding matrix with given `string` data.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

import numpy as np

# To tokenise input base.
BASE = {
    True: {
        "-": 0,
        "A": 1,
        "U": 2,
        "G": 3,
        "C": 4
    },
    False: {
        "-": np.array([0, 0, 0, 0]),
        "A": np.array([1, 0, 0, 0]),
        "U": np.array([0, 1, 0, 0]),
        "G": np.array([0, 0, 1, 0]),
        "C": np.array([0, 0, 0, 1])
    }
}

# To tokenise input secondary structure position.
PAIRED = {
    True: {
        "-": 0,
        "(": 1,
        ".": 2,
        ")": 3
    },
    False: {
        "-": np.array([0, 0, 0]),
        "(": np.array([1, 0, 0]),
        ".": np.array([0, 1, 0]),
        ")": np.array([0, 0, 1])
    }
}

# To tokenise input secondary structure type.
LOOP = {
    True: {
        "-": 0,
        "E": 1,
        "B": 2,
        "M": 3,
        "X": 4,
        "I": 5,
        "H": 6,
        "S": 7
    },
    False: {
        "-": np.array([0, 0, 0, 0, 0, 0, 0]),
        "E": np.array([1, 0, 0, 0, 0, 0, 0]),
        "B": np.array([0, 1, 0, 0, 0, 0, 0]),
        "M": np.array([0, 0, 1, 0, 0, 0, 0]),
        "X": np.array([0, 0, 0, 1, 0, 0, 0]),
        "I": np.array([0, 0, 0, 0, 1, 0, 0]),
        "H": np.array([0, 0, 0, 0, 0, 1, 0]),
        "S": np.array([0, 0, 0, 0, 0, 0, 1])
    }
}


def hot_encoding(dataset: np.array(str), encoder: "dict[str, int]",
                 is_tokenise: bool = True) -> np.array(np.array(int)):
    """Modify a numpy array of string letters into a one with string digits.

    Parameters
    ----------
    dataset : np.array(str)
        The array to modify.
    encoder : dict[str, str]
        Dictionary containing which characters to modify (`key`) with what
        (`value`).
    is_tokenise : bool
        Which type of encoder to use. If set to `True`, using the tokeniser.
        Else, if set to `False`, using the array-tokeniser. By default, `True`.

    Returns
    -------
    np.array(np.array(int))
        The modify array.
    """
    # Break string into list, with 1 character per index.
    brk_array: list[list[str]] = [*dataset]
    int_array: list[list[int]] = np.array([])

    # Parsing all array's row.
    for row in brk_array:
        trlt_base: list[int] = []

        # Parsing all row's base.
        for base in row:
            trlt_base += [encoder[is_tokenise][base]]

        if int_array.shape[0] == 0:
            int_array = np.array([trlt_base])
        else:
            int_array = np.concatenate((int_array, np.array([trlt_base])),
                                       axis=0)

    # Returning into a numpy array.
    return int_array


def positional_embedding(length: int, seq_len: int, is_tokenise: bool = True,
                         len_tok_arr: int = 0) -> np.array(np.array(float)):
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
    if is_tokenise:
        # Create an empty array.
        pos_emb: np.array = np.zeros([length, seq_len])
        # Create a array of `length` size.
        dimension: np.array = np.transpose(np.array([range(length)] * seq_len))
        # Create a array of `seq_len` size.
        to_filter: np.array = np.array(range(seq_len))

        # To filter odd and even index.
        o_f: np.array = to_filter % 2 == 0
        e_f: np.array = to_filter % 2 == 1

        # Doing arithmetic operation applied with a vectoriel method.
        pos_emb[:, o_f] = np.sin(np.divide(dimension[:, o_f], 10_000 **
                                           (2 * to_filter[o_f] / seq_len)))
        pos_emb[:, e_f] = np.cos(np.divide(dimension[:, e_f], 10_000 **
                                           (2 * to_filter[e_f] / seq_len)))
    else:
        # Create an empty array.
        pos_emb: np.array = np.zeros([length, seq_len, len_tok_arr])
        # Create a array of `length * seq_len * len_tok_arr` size.
        dimension: np.array = np.transpose(np.array([[range(length)] * seq_len]
                                                    * len_tok_arr))
        # Create a array of `seq_len * len_tok_arr` size.
        to_filter: np.array = np.transpose(np.array([range(seq_len)]
                                                    * len_tok_arr))

        # To filter odd and even index.
        o_f: np.array = (to_filter % 2 == 0)[:, 1]
        e_f: np.array = (to_filter % 2 == 1)[:, 1]

        # Doing arithmetic operation applied with a vectoriel method.
        pos_emb[:, o_f, :] = np.sin(
            np.divide(dimension[:, o_f, ], 10_000 ** (2 * to_filter[o_f, :]
                                                      / seq_len)))
        pos_emb[:, e_f, :] = np.cos(
            np.divide(dimension[:, e_f, ], 10_000 ** (2 * to_filter[e_f, :]
                                                      / seq_len)))

    return pos_emb


if __name__ == "__main__":
    # ================
    #
    # DATA IMPORTATION
    #
    # ================
    data_train: np.array = np.load("data/train.npy", allow_pickle=True)

    # ======================================
    #
    # CREATING PRE-EMBEDDING MATRIX TOKENISE
    #
    # ======================================
    # Creating tokenise embedding matrix.
    sequence: np.array = hot_encoding(dataset=data_train[:, 1], encoder=BASE)
    second_strct: np.array = hot_encoding(dataset=data_train[:, 2],
                                          encoder=PAIRED)
    loop_type: np.array = hot_encoding(dataset=data_train[:, 3], encoder=LOOP)
    
    # Creating positional embedding matrix.
    dim: tuple = sequence.shape
    pos: np.array = positional_embedding(length=dim[0], seq_len=dim[1])

    # Creating the input embedding matrix.
    tok_emb_seq: np.array = sequence + pos
    tok_emb_sec: np.array = second_strct + pos
    tok_emb_loop: np.array = loop_type + pos
    
    print(tok_emb_seq)

    # =============================
    #
    # CREATING PRE-EMBEDDING MATRIX
    #
    # =============================
    # Creating tokenise embedding matrix.
    sequence: np.array = hot_encoding(dataset=data_train[:, 1], encoder=BASE,
                                      is_tokenise=False)
    second_strct: np.array = hot_encoding(dataset=data_train[:, 2],
                                          encoder=PAIRED, is_tokenise=False)
    loop_type: np.array = hot_encoding(dataset=data_train[:, 3], encoder=LOOP,
                                       is_tokenise=False)

    # Creating positional embedding matrix.
    dim_seq: tuple = sequence.shape
    pos_seq: np.array = positional_embedding(
        length=dim_seq[0],
        seq_len=dim_seq[1],
        is_tokenise=False,
        len_tok_arr=dim_seq[2]
    )
    dim_sec: tuple = second_strct.shape
    pos_sec: np.array = positional_embedding(
        length=dim_sec[0],
        seq_len=dim_sec[1],
        is_tokenise=False,
        len_tok_arr=dim_sec[2]
    )
    dim_loop: tuple = loop_type.shape
    pos_loop: np.array = positional_embedding(
        length=dim_loop[0],
        seq_len=dim_loop[1],
        is_tokenise=False,
        len_tok_arr=dim_loop[2]
    )

    # Creating the input embedding matrix.
    emb_seq: np.array = sequence + pos_seq
    emb_sec: np.array = second_strct + pos_sec
    emb_loop: np.array = loop_type + pos_loop
