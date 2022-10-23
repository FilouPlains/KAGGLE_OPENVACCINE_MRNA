"""Main program to launch the neural network learning.
"""

__authors__ = ["BEL Alexis", "BELAKTIB Anas", "OUSSAREN Mohamed",
               "ROUAUD Lucas"]
__contact__ = ["alexbel28@yahoo.fr", "anas.belaktib@etu.u-paris.fr",
               "oussarenmohamed@outlook.fr", "lucas.rouaud@gmail.com"]
__date__ = "23/09/2022"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# Importation of other python module.
import argparse
import os
import sys

from textwrap import wrap


def parsing():
    """This function call the parser to get all necessary program's arguments.

    Returns
    -------
    dict[str, val**]
        Permit the accessibility to access to all given arguments with their
        values, thanks to a dictionary.
    """
    # ==================
    #
    # CREATE THE PARSER
    #
    # ==================
    # Setup the arguments parser object.
    parser = argparse.ArgumentParser()

    # Description of the program given when the help is cast.
    DESCRIPTION = ("Give this program input data to train a neural network, or "
                   "use a already existing neural network to predict other "
                   "data.")

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # ==========
    #
    # ARGUMENTS
    #
    # ==========
    # == REQUIRED.
    parser.add_argument(
        "-i, --input",
        required=True,
        dest="input",
        type=str,
        help="Data to train the neural network or to predict values."
    )
    parser.add_argument(
        "-o, --output",
        required=True,
        dest="output",
        type=str,
        help="In function of given parameters, output a train neural network "
        "or predict data."
    )
    parser.add_argument(
        "--pred, --predict_data",
        required=False,
        dest="predict_data",
        type=str,
        help=("An optional argument. If used, the program try to predict `Y` "
              "data based on input `X` data. Else, the program train a neural "
              "network. By default, not set, so the program train a neural "
              "network. When this option is used, give after the `npy` file to "
              "use to predict `Y` data.")
    )

    # == OPTIONAL.
    # Which neural network to use.
    parser.add_argument(
        "--cnn",
        required=False,
        dest="cnn",
        action="store_true",
        help=("An optional argument, used by default. If used, the neural "
              "network to train will be a `CNN` one.")
    )
    parser.add_argument(
        "--inc, --inception",
        required=False,
        dest="inception",
        action="store_true",
        help=("An optional argument, NOT used by default. If used, the neural "
              "network to train will be a `inception` one.")
    )
    # Which embedding input to use.
    parser.add_argument(
        "--ke, --keras_embedding",
        required=False,
        dest="keras_embedding",
        action="store_true",
        help=("An optional argument, used by default. If used, the input is "
              "transform as a keras embedding.")
    )
    parser.add_argument(
        "--owe, --own_embedding",
        required=False,
        dest="own_embedding",
        action="store_true",
        help=("An optional argument, NOT used by default. If used, the input "
              "is transform as a embedding create by our own way.")
    )
    parser.add_argument(
        "--re, --rnabert_embedding",
        required=False,
        dest="rnabert_embedding",
        action="store_true",
        help=("An optional argument, NOT used by default. If used, the input "
              "is transform as a embedding output by RNABERT.")
    )

    # Transform the input into a dictionary with arguments as key.
    argument = vars(parser.parse_args())

    # ===============================
    #
    # TESTS IF PARAMETERS ARE CORRECT
    #
    # ===============================
    # Checking input/output file are existing.

    if not os.path.exists(argument["input"]):
        sys.exit(f"\n[Err## 1] The input file '{argument['input']}' does not "
                 "exist. Please check this given file.")
    elif os.path.exists(argument["output"]):
        sys.exit(f"\n[Err## 2] The output file '{argument['output']}' does "
                 "exist. Please change the output name file.")

    predict_data = argument["predict_data"] is None

    # In neural network training mode.
    if predict_data:
        print("=" * 80 + "\n")
        line = ("- Parameters '-pred' or '--predict_data' *NOT* given. The "
                "program will try, with input data, to train a neural "
                "network.")
        print("\n  ".join(wrap(line, 80)))
        line = ("- /!\\ WARNING: Choose the same input embedding method used "
                "to train the previous neural network.")
        print("\n  ".join(wrap(line, 80)))
        print("\n" + "=" * 80 + "\n")

        # Setting extensions to check.
        input_extension = ["npy"]
        output_extension = ["json"]

        # Checking which neural network is select (set default choice and check
        # than only one neural network is ask).
        if argument["cnn"] + argument["inception"] > 1:
            sys.exit("\n[Err## 6] Give too much embedding parameters. Please, "
                     "select only one.\n\t - [-oe, --own_embedding]: "
                     f"{argument['cnn']}\n\t - [-re, --rnabert_embedding]: "
                     f"{argument['inception']}\n")
        elif argument["cnn"] + argument["inception"] == 0:
            print("=" * 80 + "\n")
            print("- No network given, default one 'CNN' select.\n")
            print("=" * 80 + "\n")

            argument["cnn"] = True
    # In predict `Y` values mode.
    else:
        print("=" * 80 + "\n")
        line = ("- Parameters '-pred' or '--predict_data' given. The program "
                "will try, with an input neural network, to predict 'Y' "
                "data.\n")
        print("\n  ".join(wrap(line, 80)))
        print("\n" + "=" * 80 + "\n")

        # Setting extensions to check.
        input_extension = ["json"]
        output_extension = ["npy", "csv", "tsv"]

        prediction_ext = argument["predict_data"].split(".")[-1]

        # Checking data to predict's file.
        if prediction_ext not in ["npy"]:
            sys.exit(f"\n[Err## 3] The prediction extension '.{prediction_ext}'"
                     " isn't a valid one. Please change it. Valid extensions "
                     f"are:\n{['npy']}")

    input_ext = argument["input"].split(".")[-1]
    output_ext = argument["output"].split(".")[-1]

    # Checking file extension.
    if input_ext not in input_extension:
        sys.exit(f"\n[Err## 4] The input extension '.{input_ext}' isn't a "
                 "valid one. Please change it. Valid extensions are:\n"
                 f"{input_extension}")
    if output_ext not in output_extension:
        sys.exit(f"\n[Err## 5] The output extension '.{output_ext}' isn't a "
                 "valid one. Please change it. Valid extensions are:\n"
                 f"{output_extension}")

    # Checking if embedding input method are correctly given.
    if argument["keras_embedding"] + argument["own_embedding"] + \
            argument["rnabert_embedding"] > 1:
        sys.exit("\n[Err## 6] Give too much embedding parameters. Please, "
                 "select only one.\n\t- [-ke, --keras_embedding]: "
                 f"{argument['keras_embedding']}\n\t- [-oe, --own_embedding]: "
                 f"{argument['own_embedding']}\n\t- [-re, --rnabert_embedding]"
                 f": {argument['rnabert_embedding']}\n")
    elif argument["keras_embedding"] + argument["own_embedding"] + \
            argument["rnabert_embedding"] == 0:
        print("=" * 80 + "\n")
        print("- No input embedding specified, default one 'keras embedding' "
              "select.\n")
        print("=" * 80 + "\n")

        argument["keras_embedding"] = True

    return argument


if __name__ == "__main__":
    arguments = parsing()
