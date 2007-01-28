// $COPYRIGHT$

#ifndef MTL_ADD_RVALUE_INCLUDE
#define MTL_ADD_RVALUE_INCLUDE

#include <boost/mpl/if.hpp>

namespace mtl {


  // Unknow (matrix) types are themselves
template <typename Matrix>
struct add_rvalue
{
    typedef Matrix     type;
};


namespace detail {
    template <typename Para>
    struct add_rvalue_parameters
    {
	// If matrix is stored on stack it can't be a rvalue
	typedef boost::if_c<
	    Para::on_stack
	  , Para
	  , matrix_parameters<typename Para::orientation, typename Para::index, typename Para::dimensions, false, true>
	>::type type;
    };
}


template <typename Value, typename Para>
struct add_rvalue<dense2D<Value, Para> >
{
    typedef dense2D<Value, typename detail::add_rvalue_parameters<Para>::type>   type;
}

template <typename Value, typename Para>
struct add_rvalue<compressed2D<Value, Para> >
{
    typedef compressed2D<Value, typename detail::add_rvalue_parameters<Para>::type>   type;
}


template <typename Value, unsigned long Mask, typename Para>
struct add_rvalue<morton_dense<Value, Mask, Para> >
{
    typedef morton_dense<Value, Mask, typename detail::add_rvalue_parameters<Para>::type>   type;
}

#if 0

// How it could be used

template <typename Matrix, typename Matrix>
typename add_rvalue_parameters<Matrix>::type
inline operator* (const Matrix& a, const Matrix& b)
{
    typename add_rvalue_parameters<Matrix>::type c;
    matmat_mult(a, b, c);
    return c;
}


#endif 


} // namespace mtl

#endif // MTL_ADD_RVALUE_INCLUDE
