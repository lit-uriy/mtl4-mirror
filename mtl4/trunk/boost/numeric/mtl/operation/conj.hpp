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

#ifndef MTL_CONJ_INCLUDE
#define MTL_CONJ_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/enable_if.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/matrix/map_view.hpp>
#include <boost/numeric/mtl/vector/map_view.hpp>

#include <complex>

namespace mtl {

namespace sfunctor {

    template <typename Value, typename AlgebraicCategory>
    struct conj
    {
	typedef Value result_type;

	static inline result_type apply(const Value& v)
	{
	    return v;
	}

	result_type operator() (const Value& v) const
	{
	    return v;
	}
    };


    template <typename Value, typename AlgebraicCategory>
    struct conj<std::complex<Value>, AlgebraicCategory>
    {
	typedef std::complex<Value> result_type;

	static inline result_type apply(const std::complex<Value>& v)
	{
	    return std::conj(v);
	}

	result_type operator() (const std::complex<Value>& v) const
	{
	    return std::conj(v);
	}
    };

    template <typename Matrix>
    struct conj<Matrix, tag::matrix>
    {
	typedef matrix::conj_view<Matrix> result_type;

	static inline result_type apply(const Matrix& matrix)
	{
	    return result_type(matrix);
	}

	result_type operator() (const Matrix& matrix) const
	{
	    return apply(matrix);
	}
    };

    template <typename Vector>
    struct conj<Vector, tag::vector>
    {
		typedef mtl::vector::conj_view<Vector> result_type;

	static inline result_type apply(const Vector& vector)
	{
	    return result_type(vector);
	}

	result_type operator() (const Vector& vector) const
	{
	    return apply(vector);
	}
    };


} // namespace sfunctor
    
    namespace vector {

	template <typename Vector>
	typename mtl::traits::enable_if_vector<Vector, conj_view<Vector> >::type
	inline conj(const Vector& v)
	{
	    return conj_view<Vector>(v);
	}
    } 

    namespace matrix {

	template <typename Matrix>
	typename mtl::traits::enable_if_matrix<Matrix, conj_view<Matrix> >::type
	inline conj(const Matrix& v)
	{
	    return conj_view<Matrix>(v);
	}
    } 

    namespace scalar {

	// Only scalar values remain here
	template <typename Value>
	typename mtl::traits::enable_if_scalar<
	    Value
	  , typename sfunctor::conj<
	        Value
			, typename mtl::traits::algebraic_category<Value>::type
	    >::result_type
	>::type
	inline conj(const Value& v)
	{
		return mtl::sfunctor::conj<Value, typename traits::algebraic_category<Value>::type>::apply(v);
	};

	float inline conj(float v) { return v; }
	double inline conj(double v) { return v; }
	long double inline conj(long double v) { return v; }
    }

    using vector::conj;
    using matrix::conj;
    using scalar::conj;


namespace sfunctor {

    template <typename Value>
    struct real
    {
	typedef Value result_type;

	static inline Value apply(const Value& v)
	{
	    return v;
	}
    };

    template <typename Value>
    struct real<std::complex<Value> >
    {
	typedef std::complex<Value> result_type;

	static inline result_type apply(const std::complex<Value>& v)
	{
	    return std::real(v);
	}
    };
}

template <typename Value>
inline typename sfunctor::real<Value>::result_type real(const Value& v)
{
	return sfunctor::real<Value>::apply(v);
};


namespace sfunctor {

    template <typename Value>
    struct imag
    {
	typedef Value result_type;

	static inline Value apply(const Value& v)
	{
	    using math::zero;
	    return zero(v);
	}
    };

    template <typename Value>
    struct imag<std::complex<Value> >
    {
	typedef std::complex<Value> result_type;

	static inline std::complex<Value> apply(const std::complex<Value>& v)
	{
	    return std::imag(v);
	}
    };

}

template <typename Value>
inline typename sfunctor::imag<Value>::result_type imag(const Value& v)
{
    return sfunctor::imag<Value>::apply(v);
};


} // namespace mtl

#endif // MTL_CONJ_INCLUDE
