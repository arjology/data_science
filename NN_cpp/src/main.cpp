#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <map>

#include "Network/Network.h"
#include "ProgressBar.h"

using namespace std;

#define _DEBUG

// Helper functions
double stepFunction(double x)
{
    return x > 0.9 ? 1.0 : x < 0.1 ? 0.0 : x;
}

map<int, double> find_values(vector<double> &positions, vector<int> &values)
{
  map<int, double> found_values;
  for (int i=0; i<positions.size(); i++)
  {
    if (positions[i] > 0)
    {
      found_values.insert(make_pair(values[i], (double)positions[i]));
    }
  }
  return found_values;
}

void printValues(map<int, double> &results)
{
    std::map<int, double>::iterator it = results.begin();
    while(it != results.end())
    {
      double pct = (double)100.0*it->second;
      cout << it->first << " :: " << ceil(pct* 100.0)/100.0 << "%" << endl;
        it++;
    }
}

void loadTraining(const char *filename, vector<vector<double> > &input, vector<vector<double> > &output)
{
    int trainingSize = 946;
    input.resize(trainingSize);
    output.resize(trainingSize);

    ifstream file(filename);
    if (file)
    {
        string line;
        int n;
        for (int i=0; i<trainingSize; i++)
        {
            for (int h=0; h<32; h++)
            {
                getline(file, line);
                for (int w=0; w<32; w++)
                {
                    input[i].push_back(atoi(line.substr(w, 1).c_str()));
                }
            }
            getline(file, line);
            output[i].resize(10); // Output ranges from 0 - 9
            n = atoi(line.substr(0, 1).c_str()); // Get the number represented by the array
            output[i][n] = 1; // Set value at position to 1; other values automatically 0 due to resize
        }
    }
    file.close();
}

int main(int argc, char *argv[])
{
    // learning digit recognition (0 ... 9)
    std::vector<std::vector<double> > inputVector, outputVector;
    loadTraining("training.txt", inputVector, outputVector); // load data from file "training.txt"

    /*
         32 * 32 = 1024 input neurons
         15 hidden neurons (experimental)
         10 output neurons
         0.7 learning rate (experimental)
    */
    int pixelSize = 32;
    int inputNeurons = pixelSize*pixelSize;
    int hiddenNeurons = 15;
    int outputNeurons = 10;
    int testingSize = 10;
    int trainingSize = inputVector.size() - testingSize;
    int numIterations = 30;
    #ifdef _DEBUG
    numIterations = 3;
    #endif
    double learningRate = 0.7;
    vector<int> digit_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    cout << "Input vector size: " << inputVector.size() << " x " << inputVector[0].size() << endl;
    cout << "Numbers pixel size: " << pixelSize
             << "\nInput neurons: " << inputNeurons
             << "\nHidden neurons: " << hiddenNeurons
             << "\nOutput neurons: " << outputNeurons
             << "\nLearning rate: " << learningRate
             << "\nTraining size: " << trainingSize
             << "\nTesting size: " << testingSize
             << "\nNumber of training iterations: " << numIterations
             << endl;
    Network net({inputNeurons, hiddenNeurons, outputNeurons}, learningRate);
    // train iterations
    double pct_complete = 0.0;
    for (int i=0; i<numIterations; i++)
    {
        for (int j=0; j<inputVector.size()-10; j++) // testing on last 10 examples
        {
            net.computeOutput(inputVector[j]);
            net.learn(outputVector[j]);
        }
        pct_complete = (i+1)/(float)numIterations;
        pbar(pct_complete);
    }
    cout << endl;
    
    // start_test
    #ifdef _DEBUG // or #ifndef NDEBUG
    cout << "expected output: actual output" << endl;
    for (int j=0; j<10; j++)
    {
        cout << outputVector[0][j] << " ";
    }
    Matrix<double> rs = net.computeOutput(inputVector[0]).applyFunction(stepFunction); 
    cout << ": " << rs << endl; 
    map<int, double> actual = find_values(outputVector[0], digit_values);
    cout << "Actual values:" << endl;
    printValues(actual); 
    vector<double> pred = rs.row(0);
    map<int, double> predicted = find_values(pred, digit_values);
    cout << "Predicted values:" << endl;
    printValues(predicted);
    cout << endl;
    // end_test
    #endif

    // test 
    #ifndef _DEBUG
    for (int i=trainingSize; i<inputVector.size(); i++)
    {
        cout << "#" << i - trainingSize + 1 << ":\t";
        for (int j=0; j<testingSize; j++)
        {
            cout << outputVector[i][j] << " ";
        }
        Matrix<double> rs = net.computeOutput(inputVector[i]).applyFunction(stepFunction);
        cout << ": " << rs;

        map<int, double> actual = find_values(outputVector[i], digit_values);
        cout << "Actual values:" << endl;
        printValues(actual); 

        vector<double> pred = rs.row(0);
        map<int, double> predicted = find_values(pred, digit_values);
        cout << "Predicted values:" << endl;
        printValues(predicted);
        cout << endl;
    }
    cout << endl << "Saving parameters...";
    net.saveNetworkParams("params.txt");
    cout << "ok!" << endl;

    // net.loadNetworkParams("params.txt"); or Network net("params.txt");
    #endif
}