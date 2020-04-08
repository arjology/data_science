#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#include "Matrix.h"

using namespace std;

Matrix<double> X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
double learningRate;

double random(double x)
{
  return (double)(rand() % 10000 + 1)/10000 - 0.5;
}

double sigmoid(double x)
{
  return 1/(1+exp(-x));
}

double sigmoidePrime(double x)
{
  return exp(-x)/(pow(1+exp(-x), 2));
}

double stepFunction(double x)
{
  if(x>0.9){
    return 1.0;
  }
  if (x<0.1){
    return 0.0;
  }
  return x;
}

void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate)
{
  learningRate = rate;

  W1 = Matrix<double>(inputNeuron, hiddenNeuron);
  W2 = Matrix<double>(hiddenNeuron, outputNeuron);
  B1 = Matrix<double>(1, hiddenNeuron);
  B2 = Matrix<double>(1, outputNeuron);

  W1 = W1.applyFunction(random);
  W2 = W2.applyFunction(random);
  B1 = B1.applyFunction(random);
  B2 = B2.applyFunction(random);
}

Matrix<double> computeOutput(vector<double> input)
{
  X = Matrix<double>({input}); // row matrix
  H = X.dot(W1).add(B1).applyFunction(sigmoid);
  Y = H.dot(W2).add(B2).applyFunction(sigmoid);
  return Y;
}

void learn(vector<double> expectedOutput)
{
  Y2 = Matrix<double>({expectedOutput}); // row matrix

  // Calculating the partial derivative of J, the error, with respect to W1, B1, W2, B2
  
  // Computing gradients
  dJdB2 = Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
  dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
  dJdW2 = H.transpose().dot(dJdB2);
  dJdW1 = X.transpose().dot(dJdB1);

  // Update the weights
  W1 = W1.subtract(dJdW1.multiply(learningRate));
  W2 = W2.subtract(dJdW2.multiply(learningRate));
  B1 = B1.subtract(dJdB1.multiply(learningRate));
  B2 = B2.subtract(dJdB2.multiply(learningRate));
}

void loadTraining(const char *filename, vector<vector<double>> &input, vector<vector<double>> &output)
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
  srand(time(NULL)); // generate random weights

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
  double learningRate = 0.7;
  init(inputNeurons, hiddenNeurons, outputNeurons, learningRate);

  cout << "Input vector size: " << inputVector.size() << " x " << inputVector[0].size() << endl;

  // train 30 iterations
  for (int i=0; i<30; i++)
  {
    for (int j=0; j<inputVector.size()-10; j++) // testing on last 10 examples
    {
      computeOutput(inputVector[j]);
      learn(outputVector[j]);
    }
    cout << "#" << i + 1 << "/30" << endl;
  }

  // test
  cout << "expected output: actual output" << endl;
  for (int j=0; j<10; j++)
  {
    cout << outputVector[0][j] << " ";
  }
  cout << ": " << computeOutput(inputVector[0]).applyFunction(stepFunction) << endl; 
  // for (int i=inputVector.size()-10; inputVector.size(); i++)
  // {
  //   for (int j=0; j<10; j++)
  //   {
  //     cout << outputVector[i][j] << " ";
  //   }
  //   cout << ": " << computeOutput(inputVector[i]).applyFunction(stepFunction) << endl;
  // }
}
