#ifndef STD_TUPLE_UTILITY_HPP
#define STD_TUPLE_UTILITY_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

namespace STD_TUPLE_NS {
  namespace detail {
    template <bool B, class T, class E>
    struct ct_if {typedef T type;};
    template <class T, class E>
    struct ct_if<false,T,E> {typedef E type;};

    template <bool B, class T>
    struct enable_if {typedef T type;};
    template <class T>
    struct enable_if<false,T> {};

    template <bool B, class T>
    struct disable_if: public enable_if<!B,T> {};
  }
}

#endif // STD_TUPLE_UTILITY_HPP
