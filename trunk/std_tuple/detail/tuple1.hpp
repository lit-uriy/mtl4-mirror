#ifndef STD_TUPLE_TUPLE1_HPP
#define STD_TUPLE_TUPLE1_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"
#include "utility.hpp"

// Tuple traits for non-tuple objects

namespace STD_TUPLE_NS {
  namespace detail {
    encode_type<tuple1_derived>::type categorize_tuple(tuple1_tag*);

    template <class T> struct tuple1;
    template <class T> struct tuple1_nesting_depth;

    template <int Cat, class T>
    struct tuple1_nesting_depth_impl {
      BOOST_STATIC_CONSTANT(int, value = 0);
    };

    template <class T>
    struct tuple1_nesting_depth_impl<tuple1_derived, T> {
      BOOST_STATIC_CONSTANT(int, value =
	  (1 + tuple1_nesting_depth<typename tuple_element<0,T>::type>::value));
    };

    template <class T>
    struct tuple1_nesting_depth
      : public tuple1_nesting_depth_impl<tuple_category<T>::value, T> {};

    template <int N, class T> struct tuple1_elt {};

    template <class T>
    struct tuple1_elt<0, T> {
      typedef typename T::value_type type;
      typedef typename boost::add_reference<type>::type ref;
      typedef typename boost::add_reference<const type>::type cref;

      static ref get(T& x) {return x.value;}
      static cref cget(const T& x) {return x.value;}
    };

    template <class T>
    class tuple1: public tuple1_tag {
      T value;
      typedef T value_type;

      template <int N, class U> friend class tuple1_elt;

      public:
      tuple1(): value() {}
      tuple1(const tuple1& o): value(o.value) {}

      // Construct from member
      template <class U>
      tuple1(const U& v, typename enable_if<
	  (tuple1_nesting_depth<U>::value <= tuple1_nesting_depth<T>::value),
	  bool>::type = false): value(v) {}

      template <class U>
      tuple1(const U& v, int): value(v) {}
      
      // Construct from other 1-tuple
      template <class U>
      tuple1(const U& v, typename enable_if<
	  (tuple1_nesting_depth<U>::value > tuple1_nesting_depth<T>::value),
	  bool>::type = false): value(get<0>(v)) {}
    };

    template <class T>
    struct tuple_traits_impl<tuple1_derived, T> {
      BOOST_STATIC_CONSTANT(int, size = 1);
      template <int N>
      struct element: public tuple1_elt<N,T> {};
    };
  }
}

#endif // STD_TUPLE_TUPLE1_HPP
