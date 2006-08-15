#include <iostream>
#include <iomanip>

#include <boost/numeric/mtl/detail/masked_dilation_tables.hpp>


using namespace std;


int main(int argc, char** argv) 
{
    using mtl::dilated::masked_dilation_tables;
  
    typedef unsigned T;
    typedef masked_dilation_tables<T, 0x55555555>    Tb1;
    typedef masked_dilation_tables<T, 0x44444444>    Tb2;
    typedef masked_dilation_tables<T, 0xfff04040>    Tb3;

    Tb1   tb1a, tb1b;
    Tb2   tb2;
    Tb3   tb3;

    T     x= 15, y= 255;
  
    cout << "x: " << hex << x << " " << tb1a.to_masked(x) << ", " << tb1b.to_masked(x) 
	 << ", " << tb2.to_masked(x) << ", " << tb3.to_masked(x) << endl;
    cout << "y: " << hex << y << " " << tb1a.to_masked(y) << ", " << tb1b.to_masked(y) 
	 << ", " << tb2.to_masked(y) << ", " << tb3.to_masked(y) << endl;

    return 0;
}
