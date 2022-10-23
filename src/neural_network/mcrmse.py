"""The loss function for the neural networl.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


import tensorflow as tf


def mcrmse(y_true, y_pred):
    """Loss function

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