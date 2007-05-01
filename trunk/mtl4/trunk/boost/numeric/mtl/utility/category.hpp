// $COPYRIGHT$

#ifndef MTL_CATEGORY_INCLUDE
#define MTL_CATEGORY_INCLUDE

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>


namespace mtl { namespace traits {

/// Meta-function for categorizing MTL and external types
/** Has to be specialized for each %matrix, %vector, ...
    Extensively used for dispatching 
    @ingroup Tags
*/
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
struct category< dense_vector<T, Parameters> > 
{
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


template <typename Functor, typename Matrix> 
struct category<matrix::map_view<Functor, Matrix> >
{
    typedef typename boost::mpl::if_<
	boost::is_same<typename category<Matrix>::type, tag::dense2D>
      , tag::dense2D_map
      , typename category<Matrix>::type
    >::type type;
};

template <typename Scaling, typename Matrix>
struct category< matrix::scaled_view<Scaling, Matrix> >
    : public category< matrix::map_view<tfunctor::scale<Scaling, typename Matrix::value_type>, 
					    Matrix> >
{};

template <typename Matrix>
struct category< matrix::conj_view<Matrix> >
    : public category< matrix::map_view<sfunctor::conj<typename Matrix::value_type>, Matrix> >
{};



/// Meta-function for categorizing types into tag::scalar, tag::vector, and tag::matrix
/** Automatically derived from category 
    @ingroup Tags
*/
template <typename T>
struct algebraic_category
{
    typedef typename boost::mpl::if_<
	boost::is_base_of<tag::matrix, typename category<T>::type>
      , tag::matrix
      , typename boost::mpl::if_<
       	    boost::is_base_of<tag::vector, typename category<T>::type>
	  , tag::vector
	  , tag::scalar
	>::type
    >::type type;
};


}} // namespace mtl::traits 

#endif // MTL_CATEGORY_INCLUDE
