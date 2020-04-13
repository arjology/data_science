#!/bin/bash

cd build;
g++-9 ../src/main.cpp ../src/Network/Network.h ../src/Network/Network.cpp ../src/Matrix.h ../src/HelperFunctions.h -o main -std=c++11
./main
python3 ../data/plot_NN.py ../data/error.txt
