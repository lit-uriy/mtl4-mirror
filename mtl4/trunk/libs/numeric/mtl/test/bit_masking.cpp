// $COPYRIGHT$

#include <iostream>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/test/minimal.hpp>

using namespace mtl;
using namespace std;  


template <unsigned long Mask>
void test()
{
    printf("Mask %x, 32x32 base is row-major %i, is column-major %i, shark 2 row-major %i"
	   ", 4x4 row-major %i, column-major %i\n",
	   Mask, is_32_base_case_row_major<Mask>::value,
	   is_k_power_base_case_col_major<5, Mask>::value,
	   is_k_power_base_case_row_major_t_shark<5, 1, Mask>::value,
	   is_k_power_base_case_row_major<2, Mask>::value,
	   is_k_power_base_case_col_major<2, Mask>::value);
}

template <unsigned long Mask1, unsigned long Mask2>
void check_same_mask()
{
    printf("Mask1 %x, Mask2 %x\n", Mask1, Mask2);
    if (Mask1 != Mask2) throw "Different masks\n";
}


int test_main(int argc, char* argv[])
{
    using mtl::row_major; using mtl::col_major;

    const unsigned long morton= 0x55555555, morton_z= 0xaaaaaaaa, doppler_4_row= 0x5555555c,
	doppler_4_col= 0x55555553, doppler_32_row= 0x555557e0, doppler_32_col= 0x5555541f,
	doppler_32_row_shark_2= 0x555557c1;

    test<morton>();
    test<morton_z>();
    test<doppler_4_row>();
    test<doppler_4_col>();
    test<doppler_32_row>();
    test<doppler_32_col>();
    test<doppler_32_row_shark_2>();

    const unsigned long morton_gen= generate_mask<true, 0, row_major, 0>::value,
	morton_z_gen= generate_mask<false, 0, row_major, 0>::value,
	doppler_4_row_gen= generate_mask<true, 2, row_major, 0>::value,
	doppler_4_col_gen= generate_mask<true, 2, col_major, 0>::value,
	doppler_32_row_gen= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_gen= generate_mask<true, 5, col_major, 0>::value,
	doppler_32_row_shark_2_gen= generate_mask<true, 5, row_major, 1>::value;

    check_same_mask<morton, morton_gen>();
    check_same_mask<morton_z, morton_z_gen>();
    check_same_mask<doppler_4_row, doppler_4_row_gen>();
    check_same_mask<doppler_4_col, doppler_4_col_gen>();
    check_same_mask<doppler_32_row, doppler_32_row_gen>();
    check_same_mask<doppler_32_col, doppler_32_col_gen>();
    check_same_mask<doppler_32_row_shark_2, doppler_32_row_shark_2_gen>();

    return 0;
}
