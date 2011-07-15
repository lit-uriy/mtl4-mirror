// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MAT_VEC_MULT_INCLUDE
#define MTL_MAT_VEC_MULT_INCLUDE

#include <cassert>
// #include <iostream>
#include <boost/mpl/bool.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/is_static.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/enable_if.hpp>
#include <boost/numeric/mtl/utility/multi_tmp.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/meta_math/loop.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>


namespace mtl { namespace matrix {

namespace impl {

    template <std::size_t Index0, std::size_t Max0, std::size_t Index1, std::size_t Max1, typename Assign>
    struct fully_unroll_mat_cvec_mult
      : public meta_math::loop2<Index0, Max0, Index1, Max1>
    {
	typedef meta_math::loop2<Index0, Max0, Index1, Max1>                              base;
	typedef fully_unroll_mat_cvec_mult<base::next_index0, Max0, base::next_index1, Max1, Assign>  next_t;

	template <typename Matrix, typename VectorIn, typename VectorOut>
	static inline void apply(const Matrix& A, const VectorIn& v, VectorOut& w)
	{
	    Assign::update(w[base::index0], A[base::index0][base::index1] * v[base::index1]);
	    next_t::apply(A, v, w);
	}   
    };

    template <std::size_t Max0, std::size_t Max1, typename Assign>
    struct fully_unroll_mat_cvec_mult<Max0, Max0, Max1, Max1, Assign>
      : public meta_math::loop2<Max0, Max0, Max1, Max1>
    {
	typedef meta_math::loop2<Max0, Max0, Max1, Max1>                              base;

	template <typename Matrix, typename VectorIn, typename VectorOut>
	static inline void apply(const Matrix& A, const VectorIn& v, VectorOut& w)
	{
	    Assign::update(w[base::index0], A[base::index0][base::index1] * v[base::index1]);
	}   
    };

    struct noop
    {
	template <typename Matrix, typename VectorIn, typename VectorOut>
	static inline void apply(const Matrix& A, const VectorIn& v, VectorOut& w) {}
    };
} // impl

// Dense matrix vector multiplication with run-time matrix size
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void dense_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, boost::mpl::true_)
{
    vampir_trace<3017> tracer;
    typedef typename static_num_rows<Matrix>::type size_type;
    static const size_type rows_a= static_num_rows<Matrix>::value, cols_a= static_num_cols<Matrix>::value;

    assert(rows_a > 0 && cols_a > 0);
    // w= A[all][0] * v[0];  N.B.: 1D is unrolled by the compiler faster (at least on gcc)
    for (size_type i= 0; i < rows_a; i++) 
	Assign::first_update(w[i], A[i][0] * v[0]);
	
    // corresponds to w+= A[all][1:] * v[1:]; if necessary
    typedef impl::fully_unroll_mat_cvec_mult<1, rows_a, 2, cols_a, Assign>  f2;
    typedef typename boost::mpl::if_c<(cols_a > 1), f2, impl::noop>::type   f3;
    f3::apply(A, v, w);
}

// Dense matrix vector multiplication with run-time matrix size
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void dense_mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, boost::mpl::false_)
{
    vampir_trace<3018> tracer;
    // Naive implementation, will be moved to a functor and complemented with more efficient ones

    using math::zero; using mtl::vector::set_to_zero;
    if (mtl::vector::size(w) == 0) return;

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type value_type;
    typedef typename Collection<Matrix>::size_type     size_type;

    for (size_type i= 0; i < num_rows(A); i++) {
	value_type tmp= zero(w[i]);
	for (size_type j= 0; j < num_cols(A); j++) 
	    tmp+= A[i][j] * v[j];
	Assign::update(w[i], tmp);
    }
}

// Dense matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::dense)
{
# ifdef MTL_NOT_UNROLL_FSIZE_MAT_VEC_MULT
    boost::mpl::false_        selector;
# else
	mtl::traits::is_static<Matrix> selector;
# endif
    dense_mat_cvec_mult(A, v, w, Assign(), selector);
}

// Multi-vector vector multiplication (tag::multi_vector is derived from dense)
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::multi_vector)
{
    vampir_trace<3019> tracer;
    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(A); i++)
	Assign::update(w, A.vector(i) * v[i]);
}

// Transposed multi-vector vector multiplication (tag::transposed_multi_vector is derived from dense)
template <typename TransposedMatrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const TransposedMatrix& A, const VectorIn& v, VectorOut& w, Assign, tag::transposed_multi_vector)
{
    vampir_trace<3020> tracer;
    typename TransposedMatrix::const_ref_type B= A.ref; // Referred matrix

    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(B); i++)
	Assign::update(w[i], dot_real(B.vector(i), v));
}

// Hermitian multi-vector vector multiplication (tag::hermitian_multi_vector is derived from dense)
template <typename HermitianMatrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const HermitianMatrix& A, const VectorIn& v, VectorOut& w, Assign, tag::hermitian_multi_vector)
{
    vampir_trace<3021> tracer;
    typename HermitianMatrix::const_ref_type B= A.const_ref(); // Referred matrix

    if (Assign::init_to_zero) set_to_zero(w);
    for (unsigned i= 0; i < num_cols(B); i++)
	Assign::update(w[i], dot(B.vector(i), v));
}



// Sparse row-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::row_major)
{
    vampir_trace<3022> tracer;
    using namespace tag; 
    using mtl::traits::range_generator;  
    using math::zero;
    using mtl::vector::set_to_zero;

    typedef typename range_generator<row, Matrix>::type       a_cur_type;    
    typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            
    typename mtl::traits::col<Matrix>::type                   col_a(A); 
    typename mtl::traits::const_value<Matrix>::type           value_a(A); 

    if (Assign::init_to_zero) set_to_zero(w);

    typedef typename Collection<VectorOut>::value_type        value_type;
    a_cur_type ac= begin<row>(A), aend= end<row>(A);
    for (unsigned i= 0; ac != aend; ++ac, ++i) {
	value_type tmp= zero(w[i]);
	for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
	    tmp+= value_a(*aic) * v[col_a(*aic)];	
	Assign::update(w[i], tmp);
    }
}

#ifdef CRS_CVEC_MULT_TUNING
template <unsigned Index, unsigned BSize, typename SizeType>
struct crs_cvec_mult_block
{
    template <typename Matrix, typename VectorIn, typename CBlock, typename TBlock>
    void operator()(const Matrix& A, const VectorIn& v, const CBlock& cj, TBlock& tmp) const
    {
	for (SizeType j= cj.value; j != cj.sub.value; ++j) // cj is one index larger
	    tmp.value+= A.data[j] * v[A.indices[j]];
	sub(A, v, cj.sub, tmp.sub);
    }

    template <typename VectorOut, typename TBlock, typename Assign>
    void first_update(VectorOut& w, SizeType i, const TBlock& tmp, Assign as) const
    { 
	Assign::first_update(w[i + Index], tmp.value);
	sub.first_update(w, i, tmp.sub, as);
    }
    
    crs_cvec_mult_block<Index+1, BSize, SizeType> sub;
};


template <unsigned BSize, typename SizeType>
struct crs_cvec_mult_block<BSize, BSize, SizeType>
{
    template <typename Matrix, typename VectorIn, typename CBlock, typename TBlock>
    void operator()(const Matrix& A, const VectorIn& v, const CBlock& cj, TBlock& tmp) const
    {
	for (SizeType j= cj.value; j != cj.sub.value; ++j)// cj is one index larger
	    tmp.value+= A.data[j] * v[A.indices[j]];
    }

    template <typename VectorOut, typename TBlock, typename Assign>
    void first_update(VectorOut& w, SizeType i, const TBlock& tmp, Assign) const
    { 
	Assign::first_update(w[i + BSize], tmp.value);
    }
};


// Row-major compressed2D vector multiplication
template <unsigned BSize, typename MValue, typename MPara, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const compressed2D<MValue, MPara>& A, const VectorIn& v, VectorOut& w, Assign as, tag::row_major)
{
    vampir_trace<3122> tracer;
    using math::zero;

    typedef compressed2D<MValue, MPara>                       Matrix;
    typedef typename Collection<Matrix>::size_type            size_type; 
    typedef typename Collection<VectorOut>::value_type        value_type;

    if (size(w) == 0) return;
    const value_type z(math::zero(w[0]));

    size_type nr= num_rows(A), nrb= nr / BSize * BSize;
    for (size_type i= 0; i < nrb; i+= BSize) {
	multi_constant_from_array<0, BSize+1, size_type> cj(A.starts, i);
	multi_tmp<BSize, value_type>                     tmp(z);
	crs_cvec_mult_block<0, BSize-1, size_type>       block;

	block(A, v, cj, tmp);
	block.first_update(w, i, tmp, as);
    }

    for (size_type i= nrb; i < nr; ++i) {
	const size_type cj0= A.starts[i], cj1= A.starts[i+1];
	value_type      tmp0(z);
	for (size_type j0= cj0; j0 != cj1; ++j0)
	    tmp0+= A.data[j0] * v[A.indices[j0]];
	Assign::first_update(w[i], tmp0);
    }
}

template <typename MValue, typename MPara, typename VectorIn, typename VectorOut, typename Assign>
typename mtl::traits::enable_if_scalar<typename Collection<VectorOut>::value_type>::type
inline smat_cvec_mult(const compressed2D<MValue, MPara>& A, const VectorIn& v, VectorOut& w, Assign, tag::row_major)
{
    smat_cvec_mult<4>(A, v, w, Assign(), tag::row_major());
}
#endif


#if !defined(CRS_CVEC_MULT_NO_ACCEL) && !defined(CRS_CVEC_MULT_TUNING)
// Row-major compressed2D vector multiplication
template <typename MValue, typename MPara, typename VectorIn, typename VectorOut, typename Assign>
typename mtl::traits::enable_if_scalar<typename Collection<VectorOut>::value_type>::type
inline smat_cvec_mult(const compressed2D<MValue, MPara>& A, const VectorIn& v, VectorOut& w, Assign, tag::row_major)
{
    vampir_trace<3122> tracer;
    using math::zero;

    typedef compressed2D<MValue, MPara>                       Matrix;
    typedef typename Collection<Matrix>::size_type            size_type; 
    typedef typename Collection<VectorOut>::value_type        value_type;

    if (size(w) == 0) return;
    const value_type z(math::zero(w[0]));

    size_type nr= num_rows(A), nrb= nr / 4 * 4;
    for (size_type i= 0; i < nrb; i+= 4) {
	const size_type cj0= A.starts[i], cj1= A.starts[i+1], cj2= A.starts[i+2], 
	                cj3= A.starts[i+3], cj4= A.starts[i+4];
	value_type      tmp0(z), tmp1(z), tmp2(z), tmp3(z);
	for (size_type j0= cj0; j0 != cj1; ++j0)
	    tmp0+= A.data[j0] * v[A.indices[j0]];
	for (size_type j1= cj1; j1 != cj2; ++j1)
	    tmp1+= A.data[j1] * v[A.indices[j1]];
	for (size_type j2= cj2; j2 != cj3; ++j2)
	    tmp2+= A.data[j2] * v[A.indices[j2]];
	for (size_type j3= cj3; j3 != cj4; ++j3)
	    tmp3+= A.data[j3] * v[A.indices[j3]];

	Assign::first_update(w[i], tmp0);
	Assign::first_update(w[i+1], tmp1);
	Assign::first_update(w[i+2], tmp2);
	Assign::first_update(w[i+3], tmp3);
    }

    for (size_type i= nrb; i < nr; ++i) {
	const size_type cj0= A.starts[i], cj1= A.starts[i+1];
	value_type      tmp0(z);
	for (size_type j0= cj0; j0 != cj1; ++j0)
	    tmp0+= A.data[j0] * v[A.indices[j0]];
	Assign::first_update(w[i], tmp0);
    }
}
#endif

// Sparse column-major matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void smat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::col_major)
{
    vampir_trace<3023> tracer;
    using namespace tag; namespace traits = mtl::traits;
    using traits::range_generator;  
    using mtl::vector::set_to_zero;
    typedef typename range_generator<col, Matrix>::type       a_cur_type;             
    typedef typename range_generator<nz, a_cur_type>::type    a_icur_type;            

    typename traits::row<Matrix>::type                        row_a(A); 
    typename traits::const_value<Matrix>::type                value_a(A); 

    if (Assign::init_to_zero) set_to_zero(w);

    unsigned rv= 0; // traverse all rows of v
    for (a_cur_type ac= begin<col>(A), aend= end<col>(A); ac != aend; ++ac, ++rv) {
	typename Collection<VectorIn>::value_type    vv= v[rv]; 
	for (a_icur_type aic= begin<nz>(ac), aiend= end<nz>(ac); aic != aiend; ++aic) 
	    Assign::update(w[row_a(*aic)], value_a(*aic) * vv);
    }
}

// Sparse matrix vector multiplication
template <typename Matrix, typename VectorIn, typename VectorOut, typename Assign>
inline void mat_cvec_mult(const Matrix& A, const VectorIn& v, VectorOut& w, Assign, tag::sparse)
{
    smat_cvec_mult(A, v, w, Assign(), typename OrientedCollection<Matrix>::orientation());
}



}} // namespace mtl::matrix




#endif // MTL_MAT_VEC_MULT_INCLUDE

