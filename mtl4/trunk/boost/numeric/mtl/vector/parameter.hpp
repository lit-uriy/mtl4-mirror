// $COPYRIGHT$

#ifndef MTL_VECTOR_PARAMETERS_INCLUDE
#define MTL_VECTOR_PARAMETERS_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl { namespace vector {

// This type exist only for bundling template parameters (to reduce typing)
template <typename Orientation= col_major, 
	  bool OnStack= false,
	  bool RValue= false>
struct parameters 
{
    typedef Orientation orientation;
    static bool const   on_stack= OnStack;
    static bool const   is_rvalue= RValue;  // to enable shallow copy
};

template <typename Parameter>
struct is_row_major {};

template <bool OnStack, bool RValue>
struct is_row_major<parameters<row_major, OnStack, RValue> >
{
  static const bool value= true;
};

template <bool OnStack, bool RValue>
struct is_row_major<parameters<col_major, OnStack, RValue> >
{
  static const bool value= false;
};


}} // namespace mtl::vector

#endif // MTL_VECTOR_PARAMETERS_INCLUDE
