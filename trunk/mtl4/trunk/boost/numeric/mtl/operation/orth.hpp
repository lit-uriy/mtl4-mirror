// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_ORTH_INCLUDE
#define MTL_ORTH_INCLUDE

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/operation/size.hpp>
#include <boost/numeric/mtl/operation/dot.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>

namespace mtl {

    namespace impl {

	template <typename VVector>
	inline void orth(VVector& v, typename mtl::Collection<VVector>::size_type j, tag::vector)
	{
	    MTL_DEBUG_THROW_IF(j < 0 || j >= size(v), index_out_of_range());
	    using ::mtl::two_norm;
	    typedef typename mtl::Collection<VVector>::size_type  Size;

	    for (Size i= 0; i < j; ++i)
		v[j]-= dot(v[i], v[j]) * v[i];
	    v[j]/= two_norm(v[j]);
	}

	template <typename VVector>
	inline void orth(VVector& v, tag::vector)
	{
	    typedef typename mtl::Collection<VVector>::size_type  Size;
	    for (Size j= 0; j < size(v); ++j)
		orth(v, j, tag::vector());
	}



	template <typename VVector>
	dense2D<typename mtl::Collection
		   <typename mtl::Collection<VVector>::value_type
		   >::value_type >
	inline orthogonalize_factors(VVector& v, tag::vector)
	{
	    using ::mtl::two_norm; using math::zero;
	    typedef typename mtl::Collection<VVector>::size_type  Size;
	    typedef typename mtl::Collection<VVector>::value_type Vector;
	    typedef typename mtl::Collection<Vector>::value_type  Scalar;

	    dense2D<Scalar> tau(size(v), size(v));
	    tau= zero(Scalar());

	    for (Size j= 0; j < size(v); ++j) {
		for (Size i= 0; i < j; ++i) {
		    Scalar t= dot(v[i], v[j]) / tau[i][i];
		    tau[i][j]= t;
		    v[j]-= t * v[i];
		}
		tau[j][j]= dot(v[j], v[j]);
	    }
	    return tau;
	}

    } // impl



/*! Orthonormalize a vector of vectors.

    The outer type must be a random access collection and
    the vector type must provide a dot function. 
    For instance dense_vector<dense_vector<double> > or
    std::vector<dense_vector<std::complex<double> > > are eligible.
    It is planned to implement the function for matrices as well
    where the columns will be ortho-normalized.
**/
template <typename Value>
inline void orth(Value& value)
{
    return impl::orth(value, typename traits::category<Value>::type());
}

/*! Orthonormalize the i-th entry of a vector of vectors.

    The i-th vector is orthogonalized w.r.t. to the preceeding ones and
    consecutively normalized.
    The outer type must be a random access collection and
    the vector type must provide a dot function. 
    For instance dense_vector<dense_vector<double> > or
    std::vector<dense_vector<std::complex<double> > > are eligible.
    It is planned to implement the function for matrices as well
    where the columns will be ortho-normalized.
**/
template <typename Value>
inline void orth(Value& value, typename mtl::Collection<Value>::size_type i)
{
    return impl::orth(value, i, typename traits::category<Value>::type());
}


/*! Orthogonalize a vector of vectors.

    Opposed to orth the vectors are not normalized. 
    An upper matrix with the factors used in the orthogonalization is returned.
    The diagonal contains dot(v[i], v[i]).
    The returned factors are for instance used in bicgstab_ell.
    The outer type must be a random access collection and
    the vector type must provide a dot function. 
    For instance dense_vector<dense_vector<double> > or
    std::vector<dense_vector<std::complex<double> > > are eligible.
**/
template <typename Value>
dense2D<typename mtl::Collection
	<typename mtl::Collection<Value>::value_type
	>::value_type >
inline orthogonalize_factors(Value& v)
{
    return impl::orthogonalize_factors(v, typename traits::category<Value>::type());
}




} // namespace mtl

#endif // MTL_ORTH_INCLUDE
