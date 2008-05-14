// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

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


template <typename Functor, typename Vector> 
struct category<vector::map_view<Functor, Vector> >
    : public category<Vector>
{};

template <typename Scaling, typename Vector>
struct category< vector::scaled_view<Scaling, Vector> >
    : public category< vector::map_view<tfunctor::scale<Scaling, typename Vector::value_type>, 
					Vector> >
{};

// added by Hui Li
template <typename Vector,typename RScaling>
struct category< vector::rscaled_view<Vector,RScaling> >
    : public category< vector::map_view<tfunctor::rscale<typename Vector::value_type,RScaling>, 
					Vector> >
{};

// added by Hui Li
template <typename Vector,typename Divisor>
struct category< vector::divide_by_view<Vector,Divisor> >
    : public category< vector::map_view<tfunctor::divide_by<typename Vector::value_type,Divisor>, 
					Vector> >
{};

template <typename Vector>
struct category< vector::conj_view<Vector> >
    : public category< vector::map_view<sfunctor::conj<typename Vector::value_type>, Vector> >
{};


namespace detail {
    
    // Helper to remove unsupported techniques in views
    template <typename Matrix>
    struct simple_matrix_view_category
    {
      private:
        typedef typename boost::mpl::if_<
    	    boost::is_same<typename category<Matrix>::type, tag::dense2D>
          , tag::dense2D_view
          , typename category<Matrix>::type
	>::type tmp1;

        typedef typename boost::mpl::if_<
    	    boost::is_same<typename category<Matrix>::type, tag::morton_dense>
          , tag::morton_view
          , tmp1
	>::type tmp2;

      public:
        typedef typename boost::mpl::if_<
    	    boost::is_same<typename category<Matrix>::type, tag::compressed2D>
          , tag::compressed2D_view
          , tmp2
	>::type type;
    };

} // detail


template <typename Functor, typename Matrix> 
struct category<matrix::map_view<Functor, Matrix> >
    : public detail::simple_matrix_view_category<Matrix>
{};

template <typename Scaling, typename Matrix>
struct category< matrix::scaled_view<Scaling, Matrix> >
    : public category< matrix::map_view<tfunctor::scale<Scaling, typename Matrix::value_type>, 
					    Matrix> >
{};

// added by Hui Li
template <typename Matrix, typename RScaling>
struct category< matrix::rscaled_view<Matrix,RScaling> >
    : public category< matrix::map_view<tfunctor::rscale<typename Matrix::value_type,RScaling>, 
					Matrix> >
{};

// added by Hui Li
template <typename Matrix, typename Divisor>
struct category< matrix::divide_by_view<Matrix,Divisor> >
    : public category< matrix::map_view<tfunctor::divide_by<typename Matrix::value_type,Divisor>, 
					Matrix> >
{};

template <typename Matrix>
struct category< matrix::conj_view<Matrix> >
    : public category< matrix::map_view<sfunctor::conj<typename Matrix::value_type>, Matrix> >
{};

template <typename Matrix>
struct category< matrix::hermitian_view<Matrix> >
    : public category< matrix::map_view<sfunctor::conj<typename Matrix::value_type>, 
					transposed_view<Matrix> > >
{};

template <typename Matrix>
struct category< matrix::banded_view<Matrix> >
    : public detail::simple_matrix_view_category<Matrix>
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
