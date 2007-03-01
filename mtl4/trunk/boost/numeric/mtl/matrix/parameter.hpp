// $COPYRIGHT$

#ifndef MTL_MATRIX_PARAMETERS_INCLUDE
#define MTL_MATRIX_PARAMETERS_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/detail/index.hpp>
#include <boost/numeric/mtl/matrix/dimension.hpp>

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

template <typename Parameter>
struct is_row_major {};

template <typename Index, typename Dimensions, bool OnStack, bool RValue>
struct is_row_major<matrix_parameters<row_major, Index, Dimensions, OnStack, RValue> >
{
  static const bool value= true;
};

template <typename Index, typename Dimensions, bool OnStack, bool RValue>
struct is_row_major<matrix_parameters<col_major, Index, Dimensions, OnStack, RValue> >
{
  static const bool value= false;
};


} // namespace mtl

#endif // MTL_MATRIX_PARAMETERS_INCLUDE
