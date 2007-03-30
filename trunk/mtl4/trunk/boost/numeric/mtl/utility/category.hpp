// $COPYRIGHT$

#ifndef MTL_CATEGORY_INCLUDE
#define MTL_CATEGORY_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>


namespace mtl { namespace traits {

// Get tag for dispatching matrices, vectors, ...
// Has to be specialized for each matrix, vector, ...
template <typename Collection> struct category 
{
    typedef tag::unknown type;
};


template <typename Value, typename Parameters>
struct category<dense2D<Value, Parameters> > 
{
    typedef tag::dense2D type;
};

template <typename Elt, unsigned long BitMask, typename Parameters>
struct category<morton_dense<Elt, BitMask, Parameters> >
{
    typedef mtl::tag::morton_dense type;
};

template <typename Elt, typename Parameters>
struct category<compressed2D<Elt, Parameters> > 
{
    typedef tag::compressed2D type;
};


template <typename T, typename Parameters>
struct category< dense_vector<T, Parameters> > {
    typedef typename boost::mpl::if_<
	boost::is_same<typename Parameters::orientation, row_major>
      , tag::dense_row_vector 
      , tag::dense_col_vector 
    >::type type;
} ;


template <class E1, class E2>
struct category< vector::vec_vec_add_expr<E1,E2> >
{
    typedef category<E1> type;
};


template <class E1, class E2>
struct category< vector::vec_vec_minus_expr<E1,E2> >
{
    typedef category<E1> type;
};


}} // namespace mtl::traits 

#endif // MTL_CATEGORY_INCLUDE
