// $COPYRIGHT$
//
// Written by Jiahu Deng and Peter Gottschling

#include <iostream>
#include <boost/test/minimal.hpp>

#include <boost/numeric/mtl/detail/dilated_int.hpp>

using namespace mtl;
using namespace mtl::dilated;
using namespace std;

const unsigned EvenBits = 0x55555555;
const unsigned OddBits = 0xaaaaaaaa;
 
#define EvenPlus1(j)     (((j) -EvenBits)&EvenBits)
#define OddPlus1(j)      (((j) -OddBits)&OddBits)
#define AntiEvenPlus1(j)    (((j) - 0xffffffff) | OddBits)
#define AntiOddPlus1(j)     (((j) - 0xffffffff) | EvenBits)

struct morton_exception {};

template <typename T>
void test_plus1(T)
{
    T a(1), b(3), c(4);
    cout << "a = " << a << ", b = " << b << ", a + b = " << a + b << ", c = " << c << "\n";
    if ((a + b).i != c.i) throw morton_exception();

    cout << "c - b = " << c - b << "\n";
    if ((c - b).i != a.i) throw morton_exception();
}  
 
template <typename T> 
void test_plus2(T)
{
    T a(22), b(33), c(55);
    cout << "a = " << a << ", b = " << b << ", a + b = " << a + b << ", c = " << c << "\n";
    if ((a + b).i != c.i) throw morton_exception();

    cout << "c - b = " << c - b << "\n";
    if ((c - b).i != a.i) throw morton_exception();
}    

template <typename T>
void test_dilated(string s, T dil1, typename T::value_type exp_increment) 
{
    printf("%s %x, bit mask is %x, negated mask is %x\n", s.c_str(), dil1.i, dil1.bit_mask, dil1.anti_mask);
    printf("%x\n", (++dil1).i);
    if (dil1.i != exp_increment) throw morton_exception();

    for (int i = 1; i < 6; ++i) {
	T dilated(i);
	cout << "dilated(" << i << ") = " << dilated <<  "\n";
	if (dil1.i != dilated.i) throw morton_exception();
	++dil1;
    }
    test_plus1(dil1); test_plus2(dil1);
}
 


int test_main(int argc, char* argv[])
{
    
    dilated::dilated_int<unsigned, odd_bits<unsigned>::value, true>     dil1;
    test_dilated( "Odd normalized", dil1, 2);

    dilated::dilated_int<unsigned, even_bits<unsigned>::value, true>     dil2;
    test_dilated( "Even normalized", dil2, 1);

    dilated::dilated_int<unsigned, odd_bits<unsigned>::value, false>     dil3;
    test_dilated( "Odd anti-normalized", dil3, 0x55555557);

    dilated::dilated_int<unsigned, even_bits<unsigned>::value, false>     dil4;
    test_dilated( "Even anti-normalized", dil4, 0xaaaaaaab);

    return 0;
}

