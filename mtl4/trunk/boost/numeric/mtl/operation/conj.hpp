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
	static inline Value apply(const Value& v)
	{
	    return v;
	}
    };

    template <typename Value>
    struct conj<std::complex<Value> >
    {
	static inline std::complex<Value> apply(const std::complex<Value>& v)
	{
	    return std::conj(v);
	}
    };

}
    
template <typename Value>
Value inline conj(const Value& v)
{
    return sfunctor::conj::apply(v);
};


namespace sfunctor {

    template <typename Value>
    struct real
    {
	static inline Value apply(const Value& v)
	{
	    return v;
	}
    };

    template <typename Value>
    struct real<std::complex<Value> >
    {
	static inline std::complex<Value> apply(const std::complex<Value>& v)
	{
	    return std::real(v);
	}
    };
}

template <typename Value>
Value inline real(const Value& v)
{
    return sfunctor::real::apply(v);
};


namespace sfunctor {

    template <typename Value>
    struct imag
    {
	static inline Value apply(const Value& v)
	{
	    return v;
	}
    };

    template <typename Value>
    struct imag<std::complex<Value> >
    {
	static inline std::complex<Value> apply(const std::complex<Value>& v)
	{
	    return std::imag(v);
	}
    };

}

template <typename Value>
Value inline imag(const Value& v)
{
    return sfunctor::imag::apply(v);
};


} // namespace mtl

#endif // MTL_CONJ_INCLUDE
