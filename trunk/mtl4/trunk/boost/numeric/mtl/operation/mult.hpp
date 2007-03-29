// $COPYRIGHT$

#ifndef MTL_MULT_INCLUDE
#define MTL_MULT_INCLUDE

#include <boost/numeric/mtl/utility/traits.hpp>
#include <boost/numeric/mtl/operation/dmat_dmat_mult.hpp>
#include <boost/numeric/mtl/operation/mult_specialize.hpp>
#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/operation/mult_assign_mode.hpp>
#include <boost/mpl/if.hpp>

namespace mtl {

template <typename A, typename B, typename C>
inline void mult(const A& a, const B& b, C& c)
{
    // dispatch between matrices, vectors, and scalars
    using traits::category;
    gen_mult(a, b, c, assign::assign_sum(), typename category<A>::type(), 
	     typename category<B>::type(), typename category<C>::type());
}


// Matrix multiplication
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void gen_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::matrix, tag::matrix, tag::matrix)
{
    // dispatch between dense and sparse
    using traits::category;
    mat_mat_mult(a, b, c, Assign(), typename category<MatrixA>::type(), 
		 typename category<MatrixB>::type(), typename category<MatrixC>::type());
}


// Dense Matrix multiplication
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::dense, tag::dense, tag::dense)
{
    using assign::plus_sum; using assign::assign_sum; 

    // Construct default functor: 
    // 1. Use BLAS   if available, otherwise
    // 2. Recursive multiplication with:
    //    1. Platform optimized mult on blocks   if available, otherwise
    //    2. Tiled multiplication on blocks      if available, otherwise
    //    3. Naive multiplication on blocks
    // 3. Naive multiplication on entire matrices if recursion is not available
    static const unsigned long tiling1= detail::dmat_dmat_mult_tiling1<MatrixA, MatrixB, MatrixC>::value;
    static const unsigned long tiling2= detail::dmat_dmat_mult_tiling2<MatrixA, MatrixB, MatrixC>::value;
    typedef gen_tiling_dmat_dmat_mult_t<tiling1, tiling2, plus_sum>    tiling_mult_t;

    typedef gen_platform_dmat_dmat_mult_t<plus_sum, tiling_mult_t>     platform_mult_t;
    typedef gen_recursive_dmat_dmat_mult_t<platform_mult_t>            recursive_mult_t;
    typedef gen_blas_dmat_dmat_mult_t<assign_sum, recursive_mult_t>    default_functor_t;

    // Use user-defined functor if provided (assign mode can be arbitrary)
    typedef typename boost::mpl::if_<
	detail::dmat_dmat_mult_specialize<MatrixA, MatrixB, MatrixC>
      , typename detail::dmat_dmat_mult_specialize<MatrixA, MatrixB, MatrixC>::type
      , default_functor_t
    >::type raw_functor_type;

    // Finally substitute assign mode (consistently)
    typename assign::mult_assign_mode<raw_functor_type, Assign>::type functor;

    functor(a, b, c);
}





} // namespace mtl

#endif // MTL_MULT_INCLUDE
