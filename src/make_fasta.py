import numpy as np

# Transform the numpy in .fa to be read by RNABERT
data_test = np.load("../data/test.npy", allow_pickle=True)
data_train = np.load("../data/training.npy", allow_pickle=True)

with open("../data/test.fa", "w", encoding="utf-8") as file:
    for line in data_test:
        file.write(">")
        file.write(line[0])
        file.write("\n")
        file.write((line[1]))
        file.write("\n")


with open("../data/train.fa", "w", encoding="utf-8") as file:
    for line in data_train:
        file.write(">")
        file.write(line[0])
        file.write("\n")
        file.write((line[1]))
        file.write("\n")