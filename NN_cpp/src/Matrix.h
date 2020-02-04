#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <vector>
#include <iostream>

class Matrix
{
public:
  Matrix();
  Matrix(int height, int width);
  Matrix(std::vector<std::vector<double>> const &array);
  Matrix multiply(double const &value); // Scalar multiplication
  Matrix add(Matrix const &m) const; // Addition
  Matrix subtract(Matrix const &m) const; // Subtraction
  Matrix multiply(Matrix const &m) const; // Hadamard product
  Matrix dot(Matrix const &m) const; // Dot product
  Matrix transpose() const; // Transpose matrix
  Matrix applyFunction(double (*function)(double)) const; // Elementwise-function applied to matrix
  void print(std::ostream &flux) const; // pretty printing matrix

private:
  std::vector<std::vector<double>> array;
  int height;
  int width;

};

std::ostream& operator<<(std::ostream &flux, Matrix const &m); // Overloading << operator to print easily

#endif