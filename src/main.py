"""Main program to launch the neural network learning.

Usage
-----
    Enter in the terminal `python3 src/main.py --help` to get the detailed 
    help generate by `argparse`.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# [I]
import input_network.keras_embedding as kreb
import input_network.masking as mask
import input_network.embedding as emb
import input_network.own_embedding as oweb
# [O]
import os
# [N]
import neural_network.saving_model as save
import neural_network.cross_validation as cv
import numpy as np

# [K]
from keras import Input
from keras.layers import Add
# [P]
from parsing import parsing


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":
    arg = parsing()
    file_out = arg["output"]

    # Importing data for the training.
    if arg["predict_data"] is None:
        dataset: np.array = np.load(arg["input"], allow_pickle=True)
        masked = mask.mask(2400, 130, 5, 68)

        if arg["rnabert_embedding"] is not None:
            # Generate output data.
            COLS = [7, 9, 11, 13, 15]

            data_rnabert = np.load(arg["rnabert_embedding"], allow_pickle=True)

            data_output = data_rnabert[:, COLS].tolist()
            data_output = np.array(data_output).reshape(2400, 68, 5)

            # To make data from 107 base length to 130 base length.
            full_output = np.zeros((2400, 62, 5))

            data_output = np.concatenate((data_output, full_output), axis=1)
        else:
            # Generate output data.
            COLS = [7, 9, 11, 13, 15]

            data_output = dataset[:, COLS].tolist()
            data_output = np.array(data_output).reshape(2400, 68, 5)

            # To make data from 107 base length to 130 base length.
            full_output = np.zeros((2400, 62, 5))

            data_output = np.concatenate((data_output, full_output), axis=1)
    # Importing data for the `Y` prediction.
    else:
        model = arg["input"]
        dataset: np.array = np.load(arg["predict_data"], allow_pickle=True)


        if arg["rnabert_embedding"]:
            data_rewrite = np.zeros((3634, 130, 120))
            empty_list = [[0] * 120]

            for i, line in enumerate(dataset):
                gap = 130 - len(line)

                data_rewrite[i] = np.array(dataset[i] + empty_list * gap)

            dataset = data_rewrite
        else:
            for i, line in enumerate(dataset):
                gap = 130 - len(line[1])

                dataset[i][1] += "-" * gap
                dataset[i][2] += "-" * gap
                dataset[i][3] += "-" * gap

        masked = mask.mask_test(dataset, 3634, 130, 5)

    n_line: int = dataset.shape[0]

    # Setting the embedding input.
    if arg["keras_embedding"]:
        inputs, original = kreb.keras_embedding(120)

        # Modifying data input.
        data_input = kreb.concat_data(dataset)

        if arg["predict_data"] is None:
            data_input = mask.format_input(data_input, n_line, 3)
    elif arg["own_embedding"]:
        input_seq, orig_seq = oweb.normalize_input_shape((130, 4), 120)
        input_sec, orig_sec = oweb.normalize_input_shape((130, 3), 120)
        input_loop, orig_loop = oweb.normalize_input_shape((130, 7), 120)

        inputs = Add()([input_seq, input_sec, input_loop])
        original = [orig_seq, orig_sec, orig_loop]

        # Modifying data unput.
        emb_seq = oweb.input_embedding(dataset[:, 1], emb.BASE)
        emb_sec = oweb.input_embedding(dataset[:, 2], emb.PAIRED)
        emb_loop = oweb.input_embedding(dataset[:, 3], emb.LOOP)

        if arg["predict_data"] is None:
            emb_seq = mask.format_input(emb_seq, n_line, 4)
            emb_sec = mask.format_input(emb_sec, n_line, 3)
            emb_loop = mask.format_input(emb_loop, n_line, 7)

        data_input = [emb_seq, emb_sec, emb_loop]
    elif arg["rnabert_embedding"]:
        inputs = Input(shape=(130, 120))
        original = [inputs]
        if arg["predict_data"] is None:
            data_input = mask.format_input(dataset, n_line, 120)

    # Training a neural network.
    if arg["predict_data"] is None:
        model, history = cv.cross_val(arg["cnn"], inputs, original, data_input,
                                      masked, data_output, file_out)
    # Predict `Y`.
    else:
        model = save.loading_model(arg["input"])
        predict_values = model.predict([data_input, masked])
        np.save(file_out, predict_values)

        print("=" * 80 + "\n")
        print(f"File is output as {file_out}.\n")
        print("=" * 80 + "\n")
