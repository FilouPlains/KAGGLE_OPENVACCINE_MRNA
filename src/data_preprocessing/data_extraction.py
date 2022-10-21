"""Cleaning the `.json` file into a `.npy` file. Like so, data are much
easier to use.

To load created data, do like so : `data = np.load("data/test.npy")`
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"

from numpy import array, row_stack, save


def __into_array(lst_str):
    """Function which convert a line like `"['1.0', '2.0']"` into numpy array.

    Parameters
    ----------
    lst_str : str
        The string to convert.

    Returns
    -------
    numpy.array
        The numpy array resulting from the string conversion.
    """
    # Take the line, split all `,"`, split then all `,` to do a list.
    # Then convert all `str` into `float` with map then reconvert it to list
    # then numpy.Array.
    return array(list(map(float, lst_str.split(",\"")[0][1:-1].split(","))))


def write_test():
    """Function which writes the testing dataset.
    """
    with open("../../data/test.json", "r", encoding="utf-8") as file:
        data = None

        for line in file:
            # Splitting line to work on theme.
            line_lst = line.strip().split(":")

            # Getting sequence informations.
            # Getting datat by splitting on `,`.
            seq_id = line_lst[2].split(",")[0][1:-1]
            sequence = line_lst[3].split(",")[0][1:-1]
            structure = line_lst[4].split(",")[0][1:-1]
            loop_type = line_lst[5].split(",")[0][1:-1]
            noise = float(line_lst[6].split(",")[0])
            score = float(line_lst[7].split(",")[0][:-1])

            # Concatenate data.
            line_clean: array = array([seq_id, sequence, structure, loop_type,
                                       noise, score],
                                      dtype=object)

            # Initialize data or stack to the array.
            if data is None:
                data = line_clean
            else:
                data = row_stack([data, line_clean])

    # Saving all data into a `.npy` file.
    save("../data/test.npy", data)
    print("[[DONE: WRITING AND SAVING TEST DATASET]]")


def write_train():
    """Function which writes the training dataset.
    """
    with open("../../data/train.json", "r", encoding="utf-8") as file:
        data = None

        for line in file:
            # Splitting line to work on theme.
            line_lst = line.strip().split(":")

            # Getting sequence informations.
            # Getting datat by splitting on `,`.
            seq_id = line_lst[2].split(",")[0][1:-1]
            sequence = line_lst[3].split(",")[0][1:-1]
            structure = line_lst[4].split(",")[0][1:-1]
            loop_type = line_lst[5].split(",")[0][1:-1]
            noise = float(line_lst[6].split(",")[0])
            sn_filter = float(line_lst[7].split(",")[0])
            score = int(line_lst[9].split(",")[0])

            # Getting data to predict.
            # ERRORS.
            E_reactivity = __into_array(line_lst[10])
            E_mg_ph10 = __into_array(line_lst[11])
            E_ph10 = __into_array(line_lst[12])
            E_mg_c50 = __into_array(line_lst[13])
            E_c50 = __into_array(line_lst[14])
            # VALUES.
            reactivity = __into_array(line_lst[15])
            mg_ph10 = __into_array(line_lst[16])
            ph10 = __into_array(line_lst[17])
            mg_c50 = __into_array(line_lst[18])
            c50 = __into_array(line_lst[19][:-1])

            # Concatenate data.
            line_clean: array = array([seq_id, sequence, structure, loop_type,
                                       noise, sn_filter, score, reactivity,
                                       E_reactivity, mg_ph10, E_mg_ph10, ph10,
                                       E_ph10, mg_c50, E_mg_c50, c50, E_c50],
                                      dtype=object)

            # Initialize data or stack to the array.
            if data is None:
                data = line_clean
            else:
                data = row_stack([data, line_clean])

    # Saving all data into a `.npy` file.
    save("../data/training.npy", data)
    print("[[DONE: WRITING AND SAVING TRAINING DATASET]]")


if __name__ == "__main__":
    write_test()
    write_train()
