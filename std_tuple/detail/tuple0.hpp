#ifndef STD_TUPLE_TUPLE0_HPP
#define STD_TUPLE_TUPLE0_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"

// Zero-length tuple

namespace STD_TUPLE_NS {
  namespace detail {
    struct tuple0 {typedef boost::type_traits::yes_type I_am_a_std_tuple_normal_tuple_class;};

    encode_type<tuple0_derived>::type categorize_tuple(tuple0*);

    template <class T>
    struct tuple_traits_impl<tuple0_derived, T> {
      BOOST_STATIC_CONSTANT(int, size = 0);
      template <int N>
      struct element {}; // Empty, since all accesses are out-of-bounds
    };
  }
}

#endif // STD_TUPLE_TUPLE0_HPP
