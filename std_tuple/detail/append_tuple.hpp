#ifndef STD_TUPLE_APPEND_TUPLE_HPP
#define STD_TUPLE_APPEND_TUPLE_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"

// Tuple formed from appending two other tuples

namespace STD_TUPLE_NS {
  namespace detail {
    encode_type<append_tuple_derived>::type categorize_tuple(append_tuple_tag*);

    template <class T1, class T2> struct append_tuple;

    template <int Half, int Nlocal, class T> struct append_tuple_elt {};

    template <int Nlocal, class T>
    struct append_tuple_elt<0, Nlocal, T> {
      typedef typename T::part0_type::template element<Nlocal> elt;
      typedef typename elt::type type;
      typedef typename elt::ref ref;
      typedef typename elt::cref cref;

      static ref get(T& x) {return elt::get(x.part0);}
      static cref cget(const T& x) {return elt::get(x.part0);}
    };

    template <int Nlocal, class T>
    struct append_tuple_elt<1, Nlocal, T> {
      typedef typename T::part1_type::template element<Nlocal> elt;
      typedef typename elt::type type;
      typedef typename elt::ref ref;
      typedef typename elt::cref cref;

      static ref get(T& x) {return elt::get(x.part1);}
      static cref cget(const T& x) {return elt::get(x.part1);}
    };

    template <class T1, class T2>
    class append_tuple: public append_tuple_tag {
      T1 part0; T2 part1;
      typedef T1 part0_type; typedef T2 part1_type;

      template <int Half, int Nlocal, class U>
      friend class append_tuple_elt;

      public:
      append_tuple(): part0(), part1() {}
      append_tuple(const append_tuple& o): part0(o.part0), part1(o.part1) {}

      // Construct from members
      template <class U, class V>
      append_tuple(typename boost::add_reference<const U>::type u,
		   typename boost::add_reference<const V>::type v)
	: part0(u), part1(v) {}
    };

    template <class T>
    struct tuple_traits_impl<append_tuple_derived, T> {
      BOOST_STATIC_CONSTANT(int, s0 = (tuple_size<typename T::part0_type>::value));
      BOOST_STATIC_CONSTANT(int, s1 = (tuple_size<typename T::part1_type>::value));
      BOOST_STATIC_CONSTANT(int, size = (s0 + s1));

      template <int N>
      struct element
	: public append_tuple_elt<(N >= s0), (N >= s0 ? N - s0 : N),T> {};
    };
  }
}

#endif // STD_TUPLE_APPEND_TUPLE_HPP
