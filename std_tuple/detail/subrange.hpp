#ifndef STD_TUPLE_SUBRANGE_HPP
#define STD_TUPLE_SUBRANGE_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"

// Tuple formed from appending two other tuples

namespace STD_TUPLE_NS {
  namespace detail {
    encode_type<subrange_derived>::type categorize_tuple(subrange_tag*);

    template <int Nlocal, class T>
    struct subrange_elt {
      typedef typename tuple_traits<typename T::data_type>
	  ::template element<Nlocal> elt;

      typedef typename elt::type type;
      typedef typename elt::ref ref;
      typedef typename elt::cref cref;

      static ref get(T& x) {return elt::get(x.data);}
      static cref cget(const T& x) {return elt::cget(x.data);}
    };

    template <int Start, int Len, class T>
    class tuple_subrange_type: public subrange_tag {
      const T& data;
      typedef T data_type;
      BOOST_STATIC_CONSTANT(int, start = Start);
      BOOST_STATIC_CONSTANT(int, length = Len);
      template <int N, class T> friend class tuple_traits_impl;
      template <int N, class T> friend class subrange_elt;

      public:
      tuple_subrange_type(const T& x): data(x) {}
    };
      
    template <int Start, int Len, class T>
    tuple_subrange_type<Start, Len, T>
    tuple_subrange(const T& x) {
      return x;
    }

    template <class T>
    struct tuple_traits_impl<subrange_derived, T> {
      BOOST_STATIC_CONSTANT(int, size = (T::length));
      BOOST_STATIC_CONSTANT(int, start = (T::start));

      template <int N>
      struct element: public subrange_elt<N+start,T> {};
    };
  }
}

#endif // STD_TUPLE_SUBRANGE_HPP
