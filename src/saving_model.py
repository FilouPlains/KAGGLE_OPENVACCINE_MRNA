"""Save and load the fitted model.
"""
from neural_network.cnn import cnn
import input_network.own_embedding as oweb
from keras.layers import Add
from keras import Input
from tensorflow.keras.models import  model_from_json
import os

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
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # create tha path to the weight file
    output_file_b = os.path.dirname(model_file) + "/" + os.path.basename(model_file).split(".")[0] +"_weight.h5"
    # load weights into new model
    loaded_model.load_weights(output_file_b)
    print(f"Loaded model {model_file} and weight {output_file_b} from disk")
    return loaded_model

if __name__ == "__main__":

    input_seq, orig_seq = oweb.normalize_input_shape((130, 4), 120)
    input_sec, orig_sec = oweb.normalize_input_shape((130, 3), 120)
    input_loop, orig_loop = oweb.normalize_input_shape((130, 7), 120)
    m_inputs = Add()([input_seq, input_sec, input_loop])

    # Creating model 1
    model1 = cnn(m_inputs, [orig_seq, orig_sec, orig_loop], Input((130, 5)))
    # Saving model    
    saving_model(model1, "../data/model_test")
    # loading model
    model2 = loading_model("../data/model_test.json")
    