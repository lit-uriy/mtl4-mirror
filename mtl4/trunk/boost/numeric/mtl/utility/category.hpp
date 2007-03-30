// $COPYRIGHT$

#ifndef MTL_CATEGORY_INCLUDE
#define MTL_CATEGORY_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>


namespace mtl { namespace traits {

// Get tag for dispatching matrices, vectors, ...
// Has to be specialized for each matrix, vector, ...
template <class Collection> struct category 
{
    typedef tag::unknown type;
};


template <typename Value, class Parameters>
struct category<dense2D<Value, Parameters> > 
{
    typedef tag::dense2D type;
};

template <class Elt, unsigned long BitMask, class Parameters>
struct category<morton_dense<Elt, BitMask, Parameters> >
{
    typedef mtl::tag::morton_dense type;
};

template <class Elt, class Parameters>
struct category<compressed2D<Elt, Parameters> > 
{
    typedef tag::compressed2D type;
};

}} // namespace mtl::traits 

#endif // MTL_CATEGORY_INCLUDE
