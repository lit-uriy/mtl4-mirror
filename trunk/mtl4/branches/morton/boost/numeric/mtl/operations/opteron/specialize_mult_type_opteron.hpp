// $COPYRIGHT$

#ifndef MTL_SPECIALIZE_MULT_TYPE_OPTERON_INCLUDE
#define MTL_SPECIALIZE_MULT_TYPE_OPTERON_INCLUDE

#ifdef MTL_USE_OPTERON_OPTIMIZATION

#include <boost/mpl/if.hpp>
#include <boost/numeric/mtl/morton_dense.hpp>
#include <boost/numeric/mtl/recursion/bit_masking.hpp>
#include <boost/numeric/mtl/recursion/base_case_test.hpp>
#include <boost/numeric/meta_math/is_power_of_2.hpp>
#include <boost/numeric/meta_math/log_2.hpp>

namespace mtl {

struct opteron_mult_hack {};

template <unsigned long MaskA, typename PA,
	  unsigned long MaskB, typename PB,
	  unsigned long MaskC, typename PC, typename DefaultMult>
struct specialize_mult_type<morton_dense<double, MaskA, PA>, morton_dense<double, MaskB, PB>, 
			    morton_dense<double, MaskC, PC>, recursion::bound_test_static<32>, DefaultMult>
{
    static const unsigned long base_case_bits= 5, // ld of 32
	                       tooth_length = 1;

    // Check if A and C have row-major shark tooth and B col-major shark tooth
    static const bool match_a= is_k_power_base_case_row_major_t_shark<base_case_bits, tooth_length, MaskA>::value,
	              match_b= is_k_power_base_case_col_major_t_shark<base_case_bits, tooth_length, MaskB>::value,
	              match_c= is_k_power_base_case_row_major_t_shark<base_case_bits, tooth_length, MaskC>::value,
	              match_all= match_a && match_b && match_c;

    typedef typename boost::mpl::if_c<
	match_all
      , opteron_mult_hack
      , DefaultMult
    >::type type;
};

} // namespace mtl

#endif // MTL_USE_OPTERON_OPTIMIZATION

#endif // MTL_SPECIALIZE_MULT_TYPE_OPTERON_INCLUDE
