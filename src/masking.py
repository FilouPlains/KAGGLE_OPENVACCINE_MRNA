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


def masked(line, col, depth, desire_output):
    maskering = np.zeros((line, col, depth))
    maskering[:,0:desire_output,] = 1
    return maskering


if __name__ == "__main__":
    print(masked(3, 3, 5, 2))
