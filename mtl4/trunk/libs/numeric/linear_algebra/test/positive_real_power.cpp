#include <iostream>
#include <cmath>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>

#include <libs/numeric/linear_algebra/test/algebraic_functions.hpp>
#include <libs/numeric/linear_algebra/test/power.hpp>
#include <libs/numeric/linear_algebra/test/positive_real.hpp>

// User defined data types and operators

using mtl::positive_real;

// We assume that 0 and infinity is not to guarantee invertibility

struct monoid_mult : public math::mult<positive_real> {};
struct group_mult : public math::mult<positive_real> {};

namespace math {
    template<> struct identity_t<monoid_mult, positive_real> 
	: public identity_t<mult<positive_real>, positive_real> {};

    template<> struct identity_t<group_mult, positive_real> 
	: public identity_t<mult<positive_real>, positive_real> {};
    template<> struct inverse_t<group_mult, positive_real> 
	: public inverse_t<mult<positive_real>, positive_real> {};
    template<> struct is_invertible_t<group_mult, positive_real> 
	: public is_invertible_t<mult<positive_real>, positive_real> {};
}

# ifdef LA_WITH_CONCEPTS
  namespace math { 
      concept_map Monoid< monoid_mult, positive_real > {};
      concept_map Group< group_mult, positive_real > {};
  }
# endif
 

int main(int, char* []) 
{
    using mtl::power;
    using math::mult;

    positive_real          value(3.0);

    std::cout << "3.0^77 as Magma: " << power(value, 77, mult<positive_real>())  << '\n'; 

    std::cout << "3.0^77 as Monoid: " << power(value, 77, monoid_mult())  << '\n'; 
    try {
	std::cout << "3.0^-77 as Monoid: " << power(value, -77, monoid_mult())  << '\n'; 
    } catch (char const* message) {
	std::cout << message << '\n';
    }

    std::cout << "3.0^77 as Group: " << power(value, 77, group_mult())  << '\n'; 
    std::cout << "3.0^-77 as Group: " << power(value, -77, group_mult())  << '\n'; 
 
    return 0;
}

