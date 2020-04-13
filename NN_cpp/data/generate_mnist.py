#!/usr/local/bin/python3

import numpy as np
import tensorflow as tf
from tqdm import tqdm

print("Loading mnist training data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

for _fname, x, y in zip(
    ["mnist_test.txt", "mnist_train.txt"], [x_test, x_train], [y_test, y_train]
):
    N = len(x)
    h, w = x[0].shape
    print(f"Writing {N} digits ({h} x {w}) to {_fname}")
    with open(_fname, "w") as _file:
        for image_index in tqdm(range(N)):
            for i in range(h):
                _line = ""
                for j in range(w):
                    _val = int(
                        np.ceil((x[image_index].astype(np.float32) / 255.0)[i][j])
                    )
                    _line += str(_val)
                _file.write(f"{_line}\n")
            _file.write(f"{y[image_index]}\n")
