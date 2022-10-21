"""Transform numpy array into a fasta file for RNABERT.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


import numpy as np


def write_fasta(data: np.array):
    """Take a given numpy array to write a fasta file.

    Parameters
    ----------
    data : str
        The numpy array from which to write the fasta file.
    """
    with open(data, "w", encoding="utf-8") as file:
        for line in file:
            file.write(">")
            file.write(line[0])
            file.write("\n")
            file.write(line[1])
            file.write("\n")


if __name__ == "__main__":
    # Transform the numpy array in `.fa` file to be read by RNABERT.
    data_test = np.load("../../data/test.npy", allow_pickle=True)
    data_train = np.load("../../data/training.npy", allow_pickle=True)

    write_fasta(data_test)
    write_fasta(data_train)
