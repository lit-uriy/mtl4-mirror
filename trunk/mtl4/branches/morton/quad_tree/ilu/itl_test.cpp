#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>


using namespace std;

int main(int argc, char** argv) {
  

  vector<double>   x(10);
  fill(x.begin(), x.end(), 3.0);

  cout << "x: ";
  copy(x.begin(), x.end(), ostream_iterator<double>(cout, " "));
  cout << endl; 

    

  return 0;
}
