#ifndef STD_TUPLE_PAIR_HPP
#define STD_TUPLE_PAIR_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"
#include <utility>

// Interface to std::pairs as tuple-like objects

namespace STD_TUPLE_NS {
  namespace detail {
    template <class T1, class T2>
    encode_type<std_pair_derived>::type
    categorize_tuple(std::pair<T1,T2>*);

    template <int N, class T> struct pair_elt {};

    template <class T>
    struct pair_elt<0,T> {
      typedef typename T::first_type type;
      typedef type& ref;
      typedef const type& cref;
      static ref get(T& x) {return x.first;}
      static cref cget(const T& x) {return x.first;}
    };

    template <class T>
    struct pair_elt<1,T> {
      typedef typename T::second_type type;
      typedef type& ref;
      typedef const type& cref;
      static ref get(T& x) {return x.second;}
      static cref cget(const T& x) {return x.second;}
    };

    template <class T>
    struct tuple_traits_impl<std_pair_derived, T> {
      BOOST_STATIC_CONSTANT(int, size = 2);

      template <int N>
      struct element
	: public pair_elt<N,T> {};
    };
  }
}

#endif // STD_TUPLE_PAIR_HPP
