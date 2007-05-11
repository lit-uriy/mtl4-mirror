// $COPYRIGHT$

#ifndef MTL_MULT_INCLUDE
#define MTL_MULT_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/operation/dmat_dmat_mult.hpp>
#include <boost/numeric/mtl/operation/smat_smat_mult.hpp>
#include <boost/numeric/mtl/operation/smat_dmat_mult.hpp>
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


/// Dense matrix multiplication
/**  The function for dense matrix multiplication defines a default multiplication functor. 
     Alternatively the user can define his own functors for specific triplets of matrix types, 
     see detail::dmat_dmat_mult_specialize.
     The default functor for dense matrix multiplication is: 
     -# Use BLAS   if available, otherwise
     -# Recursive multiplication with:
        -# Platform optimized mult on blocks   if available, otherwise
        -# Tiled multiplication on blocks      if available, otherwise
        -# Naive multiplication on blocks
     -# Naive multiplication on entire matrices if recursion is not available
**/
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::dense, tag::dense, tag::dense)
{
    using assign::plus_sum; using assign::assign_sum; 

    static const unsigned long tiling1= detail::dmat_dmat_mult_tiling1<MatrixA, MatrixB, MatrixC>::value;
    static const unsigned long tiling2= detail::dmat_dmat_mult_tiling2<MatrixA, MatrixB, MatrixC>::value;
    typedef gen_tiling_dmat_dmat_mult_t<tiling1, tiling2, plus_sum>    tiling_mult_t;

    typedef gen_platform_dmat_dmat_mult_t<plus_sum, tiling_mult_t>     platform_mult_t;
    typedef gen_recursive_dmat_dmat_mult_t<platform_mult_t>            recursive_mult_t;
    typedef gen_blas_dmat_dmat_mult_t<assign_sum, recursive_mult_t>    default_functor_t;

    /// Use user-defined functor if provided (assign mode can be arbitrary)
    typedef typename boost::mpl::if_<
	detail::dmat_dmat_mult_specialize<MatrixA, MatrixB, MatrixC>
      , typename detail::dmat_dmat_mult_specialize<MatrixA, MatrixB, MatrixC>::type
      , default_functor_t
    >::type raw_functor_type;

    /// Finally substitute assign mode (consistently)
    typename assign::mult_assign_mode<raw_functor_type, Assign>::type functor;

    functor(a, b, c);
}

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::dense, tag::dense, tag::sparse)
{
    // This is a useless and extremely inefficient operation!!!!
    // We compute this with a dense matrix and copy the result back
    dense2D<typename Collection<MatrixC>::value_type, matrix::parameters<> > c_copy(num_rows(c), num_cols(c));
    c_copy= c;
    mat_mat_mult(a, b, c_copy, Assign(), tag::dense(), tag::dense(), tag::dense());
    c= c_copy;
}

/// Sparse matrix multiplication
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::sparse, tag::sparse, tag::sparse)
{
    smat_smat_mult(a, b, c, Assign(), typename OrientedCollection<MatrixA>::orientation(),
		   typename OrientedCollection<MatrixB>::orientation());
}

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::sparse, tag::sparse, tag::dense)
{
    // This is a useless and extremely inefficient operation!!!!
    // We compute this with a sparse matrix and copy the result back
    compressed2D<typename Collection<MatrixC>::value_type, matrix::parameters<> > c_copy(num_rows(c), num_cols(c));
    c_copy= c;
    smat_smat_mult(a, b, c_copy, Assign(), typename OrientedCollection<MatrixA>::orientation(),
		   typename OrientedCollection<MatrixB>::orientation());
    c= c_copy;
}

/// Product of sparse times dense matrix
/**  This function (specialization of mult) is intended to multiply sparse matrices with multiple matrices
     gathered into a dense matrix.  Likewise, the resulting dense matrix corresponds to multiple vectors.
     The default functor for this operation is: 
     -# Use tiled multiplication      if available, otherwise
     -# Naive multiplication 
**/
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::sparse, tag::dense, tag::dense)
{
    using assign::plus_sum; using assign::assign_sum; 
    using namespace functor;

    // static const unsigned long tiling1= detail::dmat_dmat_mult_tiling1<MatrixA, MatrixB, MatrixC>::value;

    //typedef gen_smat_dmat_mult<Assign>                         default_functor_t;
    typedef gen_tiling_smat_dmat_mult<8, Assign>                         default_functor_t;

    // Finally substitute assign mode (consistently)
    // typename assign::mult_assign_mode<raw_functor_type, Assign>::type functor;

    default_functor_t functor;
    functor(a, b, c);
}

template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::sparse, tag::dense, tag::sparse)
{
    // This is a useless and extremely inefficient operation!!!!
    // We compute this with a sparse matrix and copy the result back
    dense2D<typename Collection<MatrixC>::value_type, matrix::parameters<> > c_copy(num_rows(c), num_cols(c));
    c_copy= c;
    mat_mat_mult(a, b, c_copy, Assign(), tag::sparse(), tag::dense(), tag::dense());
    c= c_copy;
}


template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::dense, tag::sparse, tag::dense)
{
    // This is could be a usefull operation, i.e. multiplying multiple row vectors with a sparse matrix
    // Might be supported in future
    // Now we compute this with a sparse matrix as first argument
    compressed2D<typename Collection<MatrixA>::value_type, matrix::parameters<> > a_copy(num_rows(a), num_cols(a));
    a_copy= a;
    compressed2D<typename Collection<MatrixC>::value_type, matrix::parameters<> > c_copy(num_rows(c), num_cols(c));
    c_copy= c;
    mat_mat_mult(a_copy, b, c_copy, Assign(), tag::sparse(), tag::sparse(), tag::sparse());
    c= c_copy;
}



template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void mat_mat_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::dense, tag::sparse, tag::sparse)
{
    // This is not a usefull operation, because the result is dense
    // Now we compute this with a sparse matrix as first argument
    compressed2D<typename Collection<MatrixA>::value_type, matrix::parameters<> > a_copy(num_rows(a), num_cols(a));
    a_copy= a;
    mat_mat_mult(a_copy, b, c, Assign(), tag::sparse(), tag::sparse(), tag::sparse());
}


} // namespace mtl

#endif // MTL_MULT_INCLUDE
