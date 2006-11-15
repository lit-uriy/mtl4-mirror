// $COPYRIGHT$

#include <iostream>


// using namespace mtl;
using namespace std;  

 
/*
     Bit masks:

     i-order (cyrillic i):
     ---------------------
  
     binary:     0101010101 ....
     0x55555555


     z-order:
     --------
     
     binary:     1010101010 ...
     0xaaaaaaaa

     row major:
     ----------

     with 2^k columns
     binary:    111111111...1000...0
                             ------- k 0s at the end (LSB), all bits before 1s (MSB)

     column major:
     -------------

     with 2^k rows
     binary:    000000000...0111...1
                             ------- k 1s at the end (LSB), all bits before 0s (MSB)

     hybrid (Doppler):
     -----------------

     i-order
     with 2^k by 2^k base case
     row major
     binary     0101....011...10...0
                               ----- k 0s at the end (LSB); means columns
                          ----- k 1s before; means rows
                ---------- i order
     e.g. 32 by 32 base case 0101...01 11111 00000 = 0x555557e0

     column major
     binary     0101....010...01...10...0
                               ----- k 1s at the end (LSB); means columns
                          ----- k 0s before; means rows
                ---------- i order
     e.g. 32 by 32 base case 0101...01 00000 11111 = 0x5555541f

*/


// Mask for the last N bits
template <unsigned long N>
struct lsb_mask
{
    static const unsigned long value= (lsb_mask<N-1>::value << 1) | 1;
};


template <>
struct lsb_mask<0>
{
    static const unsigned long value= 0;
};


// Last N bits of Value
template <unsigned long N, unsigned long Value>
struct lsb_bits
{
    static const unsigned long value= lsb_mask<N>::value & Value;
};


// Compares to masks
template <unsigned long Mask1, unsigned long Mask2>
struct same_mask
{
    static const bool value= false;
};

template <unsigned long Mask>
struct same_mask<Mask, Mask>
{
    static const bool value= true;
};


// Row-major mask for 2^K by 2^K base case
template <unsigned long K>
struct row_major_mask
{
    static const unsigned long value= lsb_mask<K>::value << K;
};


// Column-major mask for 2^K by 2^K base case
template <unsigned long K>
struct col_major_mask
    : public lsb_mask<K>
{};


// Checks whether 2^K by 2^K base case of hybric matrix, defined by Mask, is a row-major matrix
template <unsigned long K, unsigned long Mask>
struct is_k_bit_base_case_row_major
{
    static const bool value= same_mask<lsb_bits<2*K, Mask>::value, row_major_mask<K>::value>::value;
};


// Checks whether 2^K by 2^K base case of hybric matrix, defined by Mask, is a column-major matrix
template <unsigned long K, unsigned long Mask>
struct is_k_bit_base_case_col_major
{
    static const bool value= same_mask<lsb_bits<2*K, Mask>::value, col_major_mask<K>::value>::value;
};


// Checks whether 32x32 base case of hybric matrix, defined by Mask, is a row-major matrix
template <unsigned long Mask>
struct is_32_base_case_row_major
    : public is_k_bit_base_case_row_major<5, Mask>
{};



// Checks whether 32x32 base case of hybric matrix, defined by Mask, is a col-major matrix
template <unsigned long Mask>
struct is_32_base_case_col_major
    : public is_k_bit_base_case_col_major<5, Mask>
{};


template <unsigned long Mask>
void test()
{
    printf("Mask %x, last 10 bits %x, is row major base case %i, is column-major %i"
	   ", 4x4 row-major %i, column-major %i\n",
	   Mask, lsb_bits<10, Mask>::value, is_32_base_case_row_major<Mask>::value,
	   is_k_bit_base_case_col_major<5, Mask>::value,
	   is_k_bit_base_case_row_major<2, Mask>::value,
	   is_k_bit_base_case_col_major<2, Mask>::value);
}


int main(int argc, char* argv[])
{
    const unsigned long morton= 0x55555555, morton_z= 0xaaaaaaaa, doppler_4_row= 0x5555555c,
                        doppler_4_col= 0x55555553, doppler_32_row= 0x555557e0, doppler_32_col= 0x5555541f;

    test<morton>();
    test<morton_z>();
    test<doppler_4_row>();
    test<doppler_4_col>();
    test<doppler_32_row>();
    test<doppler_32_col>();

    return 0;
}
