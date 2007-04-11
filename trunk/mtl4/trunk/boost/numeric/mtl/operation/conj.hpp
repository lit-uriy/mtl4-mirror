// $COPYRIGHT$

#ifndef MTL_CONJ_INCLUDE
#define MTL_CONJ_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

#include <complex>

namespace mtl {

namespace sfunctor {

    template <typename Value>
    struct conj
    {
	typedef Value result_type;

	static inline result_type apply(const Value& v)
	{
	    return v;
	}
    };

    template <typename Value>
    struct conj<std::complex<Value> >
    {
	typedef std::complex<Value> result_type;

	static inline result_type apply(const std::complex<Value>& v)
	{
	    return std::conj(v);
	}
    };

}
    
template <typename Value>
sfunctor::conj<Value>::result_type inline conj(const Value& v)
{
    return sfunctor::conj<Value>::apply(v);
};


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
inline sfunctor::real<Value>::result_type real(const Value& v)
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
inline sfunctor::imag<Value>::result_type imag(const Value& v)
{
    return sfunctor::imag<Value>::apply(v);
};


} // namespace mtl

#endif // MTL_CONJ_INCLUDE
