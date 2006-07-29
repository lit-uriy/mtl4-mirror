// $COPYRIGHT$

#ifndef MTL_POSITIVE_REAL_POWER_INCLUDE
#define MTL_POSITIVE_REAL_POWER_INCLUDE
 
#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>
#include <boost/numeric/linear_algebra/is_invertible.hpp>

#include <libs/numeric/linear_algebra/test/positive_real.hpp>

// User defined data types and operators

using mtl::positive_real;

struct semigroup_mult : public math::mult<positive_real> {};
struct monoid_mult : public math::mult<positive_real> {};
struct pim_mult : public math::mult<positive_real> {};     // Partially invertible monoid
struct group_mult : public math::mult<positive_real> {};

namespace math {
    template<> struct identity_t<monoid_mult, positive_real> 
	: public identity_t<mult<positive_real>, positive_real> {};

    template<> struct identity_t<pim_mult, positive_real> 
	: public identity_t<mult<positive_real>, positive_real> {};
    template<> struct inverse_t<pim_mult, positive_real> 
	: public inverse_t<mult<positive_real>, positive_real> {};
    template<> struct is_invertible_t<pim_mult, positive_real> 
	: public detail::non_zero_is_invertible_t<pim_mult, positive_real> {};

    template<> struct identity_t<group_mult, positive_real> 
	: public identity_t<mult<positive_real>, positive_real> {};
    template<> struct inverse_t<group_mult, positive_real> 
	: public inverse_t<mult<positive_real>, positive_real> {};
    template<> struct is_invertible_t<group_mult, positive_real> 
	: public is_invertible_t<mult<positive_real>, positive_real> {};
}


# ifdef LA_WITH_CONCEPTS
  namespace math { 
      concept_map SemiGroup< semigroup_mult, positive_real > {};
      concept_map Monoid< monoid_mult, positive_real > {};
      concept_map PartiallyInvertibleMonoid< pim_mult, positive_real > {};
      concept_map Group< group_mult, positive_real > {};
  }
# endif
 


#endif // MTL_POSITIVE_REAL_POWER_INCLUDE
