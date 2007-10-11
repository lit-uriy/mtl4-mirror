// $COPYRIGHT$

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/mtl.hpp>


using namespace mtl;
using namespace std;  
 

int test_main(int argc, char* argv[])
{
    dense_vector<float>    x;

    if (size(x) != 0) throw "vector should be empty";

    x.change_dim(5);
    if (size(x) != 5) throw "vector should have size 5";

    x= 3.0;
    cout << "Vector x is initialized to: " << x;

    
    x.change_dim(7);
    if (size(x) != 7) throw "vector should have size 7";

    cout << "Vector x after resizing (larger): " << x;

    x= 3.0;
    x.change_dim(4);
    if (size(x) != 4) throw "vector should have size 4";

    cout << "Vector x after resizing (smaller): " << x;

    return 0;
}
 














