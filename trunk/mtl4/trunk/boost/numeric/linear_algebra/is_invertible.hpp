// $COPYRIGHT$

#ifndef MATH_IS_INVERTIBLE_INCLUDE
#define MATH_IS_INVERTIBLE_INCLUDE

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>


namespace math {

    template <typename Op, typename Element>
    struct is_invertible 
    {
	bool operator()(const Element&) const;
    };


    // Scalars are supposed to have an additive inverse unless unsigned
    // Other types have by default no inverse (specialization needed)
    template <typename Element>
    struct is_invertible< add<Element>, Element >
    {
	bool operator() (const Element&) const 
	{
	    return boost::is_arithmetic<Element>::value 
		&& !boost::is_same<Element, unsigned char>::value 
		&& !boost::is_same<Element, unsigned short>::value 
		&& !boost::is_same<Element, unsigned int>::value 
		&& !boost::is_same<Element, unsigned long>::value 
		&& !boost::is_same<Element, unsigned long long>::value; 
	}
    };


    // shouldn't be needed !
    template <>
    struct is_invertible< mult<float>, float >
    {
	bool operator() (float v) const
	{
	    return v != 0.0f;
	}
    };


    // Default for scalars: integral are never invertible and floating points if not 0
    template <typename Element>
    struct is_invertible< mult<Element>, Element >
    {
        // typename boost::enable_if<boost::is_float<Element>, bool>::type
        bool 
	operator() (const Element& v) const 
	{
	    return v != 0.0;
	}

	typename boost::enable_if<boost::is_integral<Element>, bool>::type
	operator() (const Element&) const 
	{
	    return false;
	}    
    };

} // namespace math

#endif // MATH_IS_INVERTIBLE_INCLUDE
