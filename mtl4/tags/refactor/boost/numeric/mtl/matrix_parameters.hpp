// $COPYRIGHT$

#ifndef MTL_MATRIX_PARAMETERS_INCLUDE
#define MTL_MATRIX_PARAMETERS_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>
#include <boost/numeric/mtl/index.hpp>
#include <boost/numeric/mtl/dimensions.hpp>

namespace mtl {

// This type exist only for bundling template parameters (to reduce typing)
template <class Orientation= row_major, class Index= index::c_index,
	  class Dimensions= mtl::non_fixed::dimensions>
struct matrix_parameters 
{
    typedef Orientation orientation;
    typedef Index       index;
    typedef Dimensions  dimensions;
};



} // namespace mtl

#endif // MTL_MATRIX_PARAMETERS_INCLUDE
