#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include "op_env_setup.hpp"
#include "matrix.h"
#include "sysSolver.h" 

#include "itl_interface.hpp"


using namespace std;

int main(int argc, char** argv) {

  cout << "start\n";
  //  Matrix A;
  setup_env(mat);

  mat.printRowMaj(mat.mainNode);

  vector<double>   x(10), y(10);
  fill(x.begin(), x.end(), 3.0);

  cout << "x: ";
  copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
  cout << endl; 

  

  return 0;
}
