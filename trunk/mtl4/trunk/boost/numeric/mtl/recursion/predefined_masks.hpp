// $COPYRIGHT$

#ifndef MTL_PREDEFINED_MASKS_INCLUDE
#define MTL_PREDEFINED_MASKS_INCLUDE

#include <boost/numeric/mtl/recursion/bit_masking.hpp>

namespace mtl { namespace recursion {

    // Bitmasks: 
    const unsigned long morton_mask= generate_mask<true, 0, row_major, 0>::value,
	morton_z_mask= generate_mask<false, 0, row_major, 0>::value,
	doppler_2_row_mask= generate_mask<true, 1, row_major, 0>::value,
	doppler_2_col_mask= generate_mask<true, 1, col_major, 0>::value,
	doppler_4_row_mask= generate_mask<true, 2, row_major, 0>::value,
	doppler_4_col_mask= generate_mask<true, 2, col_major, 0>::value,
	doppler_16_row_mask= generate_mask<true, 4, row_major, 0>::value,
	doppler_16_col_mask= generate_mask<true, 4, col_major, 0>::value,
	doppler_z_16_row_mask= generate_mask<false, 4, row_major, 0>::value,
	doppler_z_16_col_mask= generate_mask<false, 4, col_major, 0>::value,
	doppler_32_row_mask= generate_mask<true, 5, row_major, 0>::value,
	doppler_32_col_mask= generate_mask<true, 5, col_major, 0>::value,
	doppler_z_32_row_mask= generate_mask<false, 5, row_major, 0>::value,
	doppler_z_32_col_mask= generate_mask<false, 5, col_major, 0>::value,
	doppler_64_row_mask= generate_mask<true, 6, row_major, 0>::value,
	doppler_64_col_mask= generate_mask<true, 6, col_major, 0>::value,
	doppler_z_64_row_mask= generate_mask<false, 6, row_major, 0>::value,
	doppler_z_64_col_mask= generate_mask<false, 6, col_major, 0>::value,
	doppler_128_row_mask= generate_mask<true, 7, row_major, 0>::value,
	doppler_128_col_mask= generate_mask<true, 7, col_major, 0>::value,
	doppler_z_128_row_mask= generate_mask<false, 7, row_major, 0>::value,
	doppler_z_128_col_mask= generate_mask<false, 7, col_major, 0>::value,
	shark_32_row_mask= generate_mask<true, 5, row_major, 1>::value,
	shark_32_col_mask= generate_mask<true, 5, col_major, 1>::value,
	shark_z_32_row_mask= generate_mask<false, 5, row_major, 1>::value,
	shark_z_32_col_mask= generate_mask<false, 5, col_major, 1>::value,
	shark_64_row_mask= generate_mask<true, 6, row_major, 1>::value,
	shark_64_col_mask= generate_mask<true, 6, col_major, 1>::value,
	shark_z_64_row_mask= generate_mask<false, 6, row_major, 1>::value,
	shark_z_64_col_mask= generate_mask<false, 6, col_major, 1>::value;


}} // namespace mtl::recursion

#endif // MTL_PREDEFINED_MASKS_INCLUDE
