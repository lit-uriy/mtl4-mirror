// $COPYRIGHT$

// File name: vector1.cpp

#include <iostream>

#include <boost/numeric/mtl/vectors.hpp>
#include <boost/numeric/mtl/operations.hpp>

using namespace mtl;
using namespace std;  

int main(int argc, char* argv[])
{
    dense_vector<double>   v(10, 0.0);

    v[7]= 3.0;

    cout << "v is " << v << "\n";

    return 0;
}
