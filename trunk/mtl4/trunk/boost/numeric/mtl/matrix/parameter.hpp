// $COPYRIGHT$

#ifndef MTL_MATRIX_PARAMETERS_INCLUDE
#define MTL_MATRIX_PARAMETERS_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/index.hpp>
#include <boost/numeric/mtl/dimensions.hpp>

namespace mtl {

// This type exist only for bundling template parameters (to reduce typing)
template <typename Orientation= row_major, 
	  typename Index= index::c_index,
	  typename Dimensions= mtl::non_fixed::dimensions,
	  bool OnStack= false,
	  bool RValue= false>
struct matrix_parameters 
{
    typedef Orientation orientation;
    typedef Index       index;
    typedef Dimensions  dimensions;
    static bool const   on_stack= OnStack;
    static bool const   is_rvalue= RValue;  // to enable shallow copy

    // Matrix dimensions must be known at compile time to be on the stack
    BOOST_STATIC_ASSERT(( !on_stack || dimensions::is_static ));
};



} // namespace mtl

#endif // MTL_MATRIX_PARAMETERS_INCLUDE
