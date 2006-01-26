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
	  bool OnHeap= false>
struct matrix_parameters 
{
    typedef Orientation orientation;
    typedef Index       index;
    typedef Dimensions  dimensions;
    static bool const   on_heap= OnHeap;
};



} // namespace mtl

#endif // MTL_MATRIX_PARAMETERS_INCLUDE
