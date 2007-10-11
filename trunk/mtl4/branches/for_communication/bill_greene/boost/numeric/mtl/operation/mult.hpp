// $COPYRIGHT$

#ifndef MTL_MULT_INCLUDE
#define MTL_MULT_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/dmat_dmat_mult.hpp>
#include <boost/numeric/mtl/operation/smat_smat_mult.hpp>
#include <boost/numeric/mtl/operation/smat_dmat_mult.hpp>
#include <boost/numeric/mtl/operation/mat_vec_mult.hpp>
#include <boost/numeric/mtl/operation/mult_specialize.hpp>
#include <boost/numeric/mtl/operation/assign_mode.hpp>
#include <boost/numeric/mtl/operation/mult_assign_mode.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/if.hpp>

namespace mtl {


/// Multiplication: mult(a, b, c) computes c= a * b; 
/** The 3 types must be compatible, e.g. all three matrices or b and c are column vectors and a is a matrix.
    The dimensions are checked at compile time. **/
template <typename A, typename B, typename C>
inline void mult(const A& a, const B& b, C& c)
{
    MTL_THROW_IF((void*)&a == (void*)&c || (void*)&b == (void*)&c, argument_result_conflict());

    // dispatch between matrices, vectors, and scalars
    using traits::category;
    gen_mult(a, b, c, assign::assign_sum(), typename category<A>::type(), 
	     typename category<B>::type(), typename category<C>::type());
}


/// Multiplication: mult_add(a, b, c) computes c+= a * b; 
/** The 3 types must be compatible, e.g. all three matrices or b and c are column vectors and a is a matrix.
    The dimensions are checked at compile time. **/
template <typename A, typename B, typename C>
inline void mult_add(const A& a, const B& b, C& c)
{
    // dispatch between matrices, vectors, and scalars
    using traits::category;
    gen_mult(a, b, c, assign::plus_sum(), typename category<A>::type(), 
	     typename category<B>::type(), typename category<C>::type());
}


/// Four term multiplication: mult(a, x, y, z) computes z= a * x + y; 
/** The 4 types must be compatible, i.e. a*x must be assignable to z and z must be incrementable by y.
    Right now, it is not more efficient than z= a * x; z+= y. For compatibility with MTL2. **/
template <typename A, typename X, typename Y, typename Z>
inline void mult(const A& a, const X& x, const Y& y, Z& z)
{
    mult(a, x, z);
    z+= y;
}


// Matrix multiplication
template <typename MatrixA, typename MatrixB, typename MatrixC, typename Assign>
inline void gen_mult(const MatrixA& a, const MatrixB& b, MatrixC& c, Assign, tag::matrix, tag::matrix, tag::matrix)
{
    MTL_THROW_IF((void*)&a == (void*)&c || (void*)&b == (void*)&c, argument_result_conflict());

    MTL_THROW_IF(num_rows(a) != num_rows(c) || num_cols(a) != num_rows(b) || num_cols(b) != num_cols(c),
		 incompatible_size());
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



// Matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void gen_mult(const Matrix& a, const VectorIn& v, VectorOut& w, Assign, tag::matrix, tag::vector, tag::vector)
{
    // Vector must be column vector
    // If vector is row vector then matrix must have one column and the operation is a outer product
    //   -> result should be a matrix too

    // Check if element types are compatible (in contrast to tag dispatching, nesting is considered here)
    BOOST_STATIC_ASSERT((boost::is_same< typename ashape::mult_op<typename ashape::ashape<Matrix>::type, 
			                                          typename ashape::ashape<VectorIn>::type >::type,
			                 ::mtl::ashape::mat_cvec_mult
			               >::value));

    MTL_THROW_IF((void*)&v == (void*)&w, argument_result_conflict());

    //MTL_THROW_IF(num_rows(a) != num_rows(w) || num_cols(a) != num_rows(v), incompatible_size());
    MTL_THROW_IF(num_rows(a) != size(w) || num_cols(a) != size(v), incompatible_size());

    // dispatch between dense and sparse matrices
    using traits::category;
    mat_cvec_mult(a, v, w, Assign(), typename category<Matrix>::type()); 
}



} // namespace mtl

#endif // MTL_MULT_INCLUDE
