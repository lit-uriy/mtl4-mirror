// $COPYRIGHT$

#ifndef MTL_VECTOR_PARAMETERS_INCLUDE
#define MTL_VECTOR_PARAMETERS_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/vector/dimension.hpp>

namespace mtl { namespace vector {

// This type exist only for bundling template parameters (to reduce typing)
template <typename Orientation= col_major, 
	  typename Dimension= non_fixed::dimension,
	  bool OnStack= false,
	  bool RValue= false>
struct parameters 
{
    typedef Orientation orientation;
    typedef Dimension   dimension;
    static bool const   on_stack= OnStack;
    static bool const   is_rvalue= RValue;  // to enable shallow copy
};


}} // namespace mtl::vector

#endif // MTL_VECTOR_PARAMETERS_INCLUDE
