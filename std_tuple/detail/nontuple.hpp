#ifndef STD_TUPLE_NONTUPLE_HPP
#define STD_TUPLE_NONTUPLE_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"

// Tuple traits for non-tuple objects

namespace STD_TUPLE_NS {
  namespace detail {
    encode_type<nontuple>::type categorize_tuple(void*);

    template <int N, class T> struct nontuple_elt {};

    template <class T>
    struct nontuple_elt<0, T> {
      typedef T type;
      typedef typename boost::add_reference<T>::type ref;
      typedef typename boost::add_reference<const T>::type cref;

      static ref get(T& x) {return x;}
      static cref cget(const T& x) {return x;}
    };

    template <class T>
    struct tuple_traits_impl<nontuple, T> {
      BOOST_STATIC_CONSTANT(int, size = 1);
      template <int N>
      struct element: public nontuple_elt<N,T> {};
    };
  }
}

#endif // STD_TUPLE_NONTUPLE_HPP
