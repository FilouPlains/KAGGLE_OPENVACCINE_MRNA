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


if __name__ == "__main__":
    print(mask(3, 3, 5, 2))
