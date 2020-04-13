#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <map>
#include <chrono>

#include "Network/Network.h"
#include "HelperFunctions.h"

using namespace std;

// #define _DEBUG

void loadTraining(const char *train_filename,
                  const char *test_filename,
                  vector<vector<double> > &input,
                  vector<vector<double> > &output,
                  int width,
                  int height,
                  int trainingSize,
                  int testingSize)
{
    int totalSize = trainingSize + testingSize;
    input.resize(totalSize);
    output.resize(totalSize);

    string line;
    int n;
    ifstream train_file(train_filename);
    if (train_file)
    {
        for (int i=0; i<trainingSize; i++)
        {
            for (int h=0; h<height; h++)
            {
                getline(train_file, line);
                for (int w=0; w<width; w++)
                {
                    input[i].push_back(atoi(line.substr(w, 1).c_str()));
                }
            }
            getline(train_file, line);
            output[i].resize(10); // Output ranges from 0 - 9
            n = atoi(line.substr(0, 1).c_str()); // Get the number represented by the array
            output[i][n] = 1; // Set value at position to 1; other values automatically 0 due to resize
        }
    }
    train_file.close();

    ifstream test_file(test_filename);
    if (test_file)
    {
        for (int i=trainingSize; i<totalSize; i++)
        {
            for (int h=0; h<height; h++)
            {
                getline(test_file, line);
                for (int w=0; w<width; w++)
                {
                    input[i].push_back(atoi(line.substr(w, 1).c_str()));
                }
            }
            getline(test_file, line);
            output[i].resize(10); // Output ranges from 0 - 9
            n = atoi(line.substr(0, 1).c_str()); // Get the number represented by the array
            output[i][n] = 1; // Set value at position to 1; other values automatically 0 due to resize
        }
    }
    test_file.close();
}

int main(int argc, char *argv[])
{ 
    #ifdef _DEBUG
    int testingSize = 2;
    int trainingSize = 10;
    int numIterations = 100;
    #endif
    #ifndef _DEBUG
    int testingSize = 10000;
    int trainingSize = 60000;
    int numIterations = 100;
    #endif
    int pixelSize = 28;
    int inputNeurons = pixelSize*pixelSize;
    int hiddenNeurons = 15;
    int outputNeurons = 10;

    vector<vector<double> > inputVector, outputVector;
    loadTraining("../data/mnist_train.txt", "../data/mnist_test.txt",
                 inputVector, outputVector,
                 pixelSize, pixelSize,
                 trainingSize, testingSize
    );
    
    double learningRate = 0.7;
    double dropoutRate = 0.4;
    vector<int> digit_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    cout << "Input vector size: " << inputVector.size() << " x " << inputVector[0].size() << endl;
    cout << "Numbers pixel size: " << pixelSize
             << "\nInput neurons: " << inputNeurons
             << "\nHidden neurons: " << hiddenNeurons
             << "\nOutput neurons: " << outputNeurons
             << "\nTraining size: " << trainingSize
             << "\nTesting size: " << testingSize
             << "\nNumber of training iterations: " << numIterations
             << "\nLearning rate: " << learningRate
             << "\nDropout rate: " << dropoutRate
             << endl;
    Network net({inputNeurons, hiddenNeurons, outputNeurons}, learningRate);
    // train iterations
    double pct_complete = 0.0;
    ofstream errorFile("../data/error.txt");
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();
    auto runtime = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    Matrix<double> rs;
    vector<double> curr_pred;
    float r;

    cout << "*** Training network..." << endl;
    for (int i=0; i<numIterations; i++)
    {
        double err = 0.0;
        for (int j=0; j<trainingSize; j++)
        {
            r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if (r < 1-dropoutRate)
            {
                rs = net.computeOutput(inputVector[j]);
                net.learn(outputVector[j]);

                if (j%100==0)
                {
                    rs = rs.applyFunction(stepFunction);
                    curr_pred = rs.row(0);
                    err += currentError(curr_pred, outputVector[j]);
                    t2 = chrono::high_resolution_clock::now();
                    runtime = chrono::duration_cast<chrono::duration<double>>(t2 - t1); 
                    pct_complete = (i*trainingSize+j+1)/(float)(numIterations*trainingSize);
                    pbar(pct_complete, err, runtime.count(), i);
                }
            }
        }
        if (i%10==0)
        {
            plotError(errorFile, i, err);
        }
    }
    errorFile.close();
    cout << endl;
    
    #ifdef _DEBUG
    // test 
    cout << "*** Inspecting predictions of testing samples..." << endl;
    for (int i=trainingSize; i<inputVector.size(); i++)
    {
        cout << "#" << i - trainingSize + 1 << ":\t";
        for (int j=0; j<10; j++)
        {
            cout << outputVector[i][j] << " ";
        }
        Matrix<double> rs = net.computeOutput(inputVector[i]).applyFunction(stepFunction);
        cout << ": " << rs;

        map<int, double> actual = findValues(outputVector[i], digit_values);
        cout << "Actual values:" << endl;
        printValues(actual); 

        vector<double> pred = rs.row(0);
        map<int, double> predicted = findValues(pred, digit_values);
        cout << "Predicted values:" << endl;
        printValues(predicted);
        cout << endl;
    }
    #endif

    #ifndef _DEBUG
    cout << "*** Inspecting predictions of testing samples..." << endl;
    int correctPred = 0;
    for (int i=trainingSize; i<inputVector.size(); i++)
    {
        Matrix<double> rs = net.computeOutput(inputVector[i]).applyFunction(stepFunction);
        vector<double> pred = rs.row(0);
        correctPred += maxValuesMatch(pred, outputVector[i]);
        pbar((double)i/(double)testingSize);
    }
    double result = (double)correctPred/(double)testingSize; 
    cout << "Testing results:\n"
         << "\t" << correctPred << "/" << testingSize << " correct"
         << endl;

    cout << endl << "Saving parameters...";
    net.saveNetworkParams("../data/params.txt");
    cout << "ok!" << endl;
    // net.loadNetworkParams("params.txt"); or Network net("params.txt");
    #endif
}