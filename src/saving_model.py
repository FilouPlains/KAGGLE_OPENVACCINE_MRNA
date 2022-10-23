"""Save and load the fitted model.
"""
from neural_network.cnn import cnn
import input_network.own_embedding as oweb
from keras.layers import Add
from keras import Input
from tensorflow.keras.models import  model_from_json
import os
from validation import non_cross_val
import numpy as np
from input_network.masking import mask, format_input
from neural_network.cnn import __mcrmse

def saving_model(model, output_file):
    """ Save a model in the folder data/model
        Parameters
    ----------
    model : model
        a model fitted (architecture, weight, compile)
    output_file : 
        Name of the folder which will contain the model
    """
    output_file_a = output_file +".json"
    output_file_b = output_file +"_weight.h5"
    # Save the weight
    model.save_weights(output_file_b)
    # Save the architecture of the model with the custom layer
    model_json = model.to_json()
    with open(output_file_a, "w") as json_file:
        json_file.write(model_json)
    print(f"Saved model {output_file_a} and weight {output_file_b} to disk")


def loading_model(model_file):
    """ Load a model from the folder data/model
        Parameters
    ----------
    model_file : 
        Name of the file which contain the model and the weight
    Return :
        Fitted model
    """
    # load the architecture of the model
    with open(model_file, 'r') as json_file:
        json_savedModel = json_file.read()
    loaded_model = model_from_json(json_savedModel)
    # create tha path to the weight file
    output_file_b = os.path.basename(model_file).split(".")[0] +"_weight.h5"
    # load weights into new model
    loaded_model.load_weights(output_file_b)
    loaded_model.compile(optimizer="adam", loss=__mcrmse)
    print(f"Loaded model {model_file} and weight {output_file_b} from disk")
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
    