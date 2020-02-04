#include <iostream>
#include <fstream>

void loadTraining(const char *filename, vector<vector<doube>> &input, vector<vector<double>> &output)
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
