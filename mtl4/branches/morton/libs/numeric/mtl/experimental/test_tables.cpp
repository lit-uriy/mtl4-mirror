#include <iostream>
#include <iomanip>

#include <boost/numeric/mtl/detail/masked_dilation_tables.hpp>


using namespace std;

using mtl::dilated::masked_dilation_tables;
  
typedef unsigned T;

typedef masked_dilation_tables<T, 0x55555555>    Tb1; 
typedef masked_dilation_tables<T, 0x44444444>    Tb2;
typedef masked_dilation_tables<T, 0xfff04040>    Tb3;


template <typename Table>
inline void check_masking(T value, T masked, Table table)
{
    if (table.to_masked(value) != masked) {
	cout << "masking " << hex << value << " with mask " << Table::mask 
	     << " should be " << masked << " but is " << table.to_masked(value) << endl;
	throw "Error masking";
    } else
	cout << "masking " << hex << value << " with mask " << Table::mask 
	     << " is " << table.to_masked(value) << endl;   
}


void test_masking(T v, T m1, T m2, T m3)
{
    check_masking(v, m1, Tb1());
    check_masking(v, m2, Tb2());
    check_masking(v, m3, Tb3());
}


template <typename Table>
inline void check_unmasking(T value, T unmasked, Table table)
{
    if (table.to_unmasked(value) != unmasked) {
	cout << "unmasking " << hex << value << " with mask " << Table::mask 
	     << " should be " << unmasked << " but is " << table.to_unmasked(value) << endl;
	throw "Error unmasking";
    } else
	cout << "unmasking " << hex << value << " with mask " << Table::mask 
	     << " is " << table.to_unmasked(value) << endl;   
    
}


void test_unmasking(T v, T um1, T um2, T um3)
{
    check_unmasking(v, um1, Tb1());
    check_unmasking(v, um2, Tb2());
    check_unmasking(v, um3, Tb3());
}



int main(int argc, char** argv) 
{
    Tb1   tb1a, tb1b;
    Tb2   tb2;
    Tb3   tb3;

    T     x= 0xf, y= 0xff;
   
    test_masking(T(0xf), T(0x55), T(0x4444), T(0x304040));
    test_masking(T(0xff), T(0x5555), T(0x44444444), T(0x3f04040));

    test_unmasking(T(0xff), T(0xf), T(3), T(1));
    test_unmasking(T(0xffff), T(0xff), T(0xf), T(3));
    test_unmasking(T(0xffffff), T(0xfff), T(0x3f), T(0x3f));
    test_unmasking(T(0xffffffff), T(0xffff), T(0xff), T(0x3fff));

    return 0;
}
