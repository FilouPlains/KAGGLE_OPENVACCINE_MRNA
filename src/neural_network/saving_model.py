"""Save and load the fitted model.
"""

# [K]
from keras import Input
from keras.models import load_model, save_model

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


def saving_model(model, output_file):
    """ Save a model in the folder data/model
    Parameters
    ----------
    model : model
        a model fitted (architecture, weight, compile)
    output_file : 
        Name of the folder which will contain the model
    """
    # Save the model. Must be a h5 file.
    save_model(model, output_file)

    print("=" * 80 + "\n")
    print(f"Saved model as {output_file}.\n")
    print("=" * 80 + "\n")


def loading_model(model_file):
    """ Load a model from the folder data/model
    Parameters
    ----------
    model_file
        Name of the file which contain the model and the weight
    Returns
    -------
    model_file
        Fitted model
    """

    # Load model only use for the prediction.
    model = load_model(model_file, compile = False)

    print("=" * 80 + "\n")
    print(f"Model {model_file} loaded.\n")
    print("=" * 80 + "\n")

    return model
