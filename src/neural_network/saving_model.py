"""Save and load the fitted model.
"""

# [I]
import input_network.own_embedding as oweb
# [N]
import numpy as np
# [O]
import os

# [K]
from keras import Input
# [N]
from neural_network.cnn import cnn
from neural_network.mcrmse import mcrmse
# [T]
from tensorflow.keras.models import model_from_json

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
    # Save the weight.
    model.save_weights(output_file)

    # Save the architecture of the model with the custom layer.
    model_json = model.to_json()

    with open(output_file, "w", encoding="utf-8") as json_file:
        json_file.write(model_json)

    print("=" * 80 + "\n")
    print(f"Saved model as {output_file}.\n")
    print("=" * 80 + "\n")


def loading_model(model_file):
    """ Load a model from the folder data/model
        Parameters
    ----------
    model_file : 
        Name of the file which contain the model and the weight
    Return :
        Fitted model
    """
    # Load the architecture of the model.
    with open(model_file, "r", encoding="utf-8") as json_file:
        json_savedModel = json_file.read()

    loaded_model = model_from_json(json_savedModel)

    # Create tha path to the weight file.
    weight_file = model_file[:-5] + ".h5"

    # Load weights into new model.
    loaded_model.load_weights(weight_file)
    loaded_model.compile(optimizer="adam", loss=mcrmse)

    print("=" * 80 + "\n")
    print(f"Model {model_file} and weight {weight_file} loaded.\n")
    print("=" * 80 + "\n")

    return loaded_model


if __name__ == "__main__":
    # Output for the 3 model
    inputs_3 = Input(shape=(130, 120))
    original_3 = inputs_3

    # Creating model 3
    model3 = cnn(inputs_3, [original_3], Input((130, 5)))
    # Saving model
    saving_model(model3, "../data/model_test")
    # loading model
    model2 = loading_model("../data/model_test.json")
    print(model2.summary())
