#ifndef DEF_PBAR
#define DEF_PBAR
#define PBWIDTH 30

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>
#include <assert.h>

using namespace std;

void pbar(double pct);
double stepFunction(double x);
map<int, double> findValues(vector<double> &positions, vector<int> &values);
void printValues(map<int, double> & results);

#endif

void pbar(double pct, double err, double runtime, int epoc)
{
    std::cout << "[";
    int pos = PBWIDTH * pct;
    std::string output;
    for (int i = 0; i < PBWIDTH; ++i)
    {
        output = i < pos ? "=" : i == pos ? ">" : " ";
        std::cout << output;
    }
    std::cout << "] "
              << int(pct * 100.0)
              << " %\t"
              << "(Epoc " << epoc << ") "
              << "\tTime: " << runtime
              << "\tError: " << err
              << "\r";
    std::cout.flush();

}

void pbar(double pct)
{
    std::cout << "[";
    int pos = PBWIDTH * pct;
    std::string output;
    for (int i = 0; i < PBWIDTH; ++i)
    {
        output = i < pos ? "=" : i == pos ? ">" : " ";
        std::cout << output;
    }
    std::cout << "] " << int(pct * 100.0) << " %\r";
    std::cout.flush();

}

double stepFunction(double x)
{
    return x > 0.9 ? 1.0 : x < 0.1 ? 0.0 : x;
}

map<int, double> findValues(vector<double> &positions, vector<int> &values)
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

double currentError(std::vector<double> &a, std::vector<double> &b)
{
  double err = 0.0;
  assert(a.size()==b.size());
  for (int idx=0; idx<a.size(); idx++)
  {
    err += std::abs(a[idx] - b[idx]);
  }
  return err;
}

int maxValuesMatch(std::vector<double> &a, std::vector<double> &b)
{
  int maxA = std::max_element(a.begin(), a.end()) - a.begin();
  int maxB = std::max_element(b.begin(), b.end()) - b.begin();
  return maxA == maxB ? 1 : 0;
}

void plotError(std::ofstream &errorFile, int iter, double err)
{
  // errorFile.write((char*)&t, sizeof(t));
  errorFile << iter << "," << err << std::endl;
}