#!/usr/local/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

if not len(sys.argv) == 2:
    print("Need an argument: error_file")
    sys.exit()

with open(sys.argv[1], "r") as error_file:
    err = error_file.readlines()
err = np.array(list(map(lambda e: list(map(float, e.strip("\n").split(","))), err)))

plt.subplots(figsize=(15, 8))
plt.plot(err[:, 0], err[:, 1], "b")
plt.savefig("../data/NN.png")
