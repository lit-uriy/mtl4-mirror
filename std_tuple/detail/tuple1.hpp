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
    template <class T> struct tuple1;
    template <class T> struct tuple1_nesting_depth;

    template <class T>
    encode_type<tuple1_derived>::type categorize_tuple(tuple1<T>*);

    template <bool IsTuple1, class T>
    struct tuple1_nesting_depth_impl {
      BOOST_STATIC_CONSTANT(int, value = 0);
    };

    template <class T>
    struct tuple1_nesting_depth_impl<true, T> {
      BOOST_STATIC_CONSTANT(int, value =
	  (1 + tuple1_nesting_depth<typename tuple_element<0,T>::type>::value));
    };

    template <class T>
    struct is_tuple1 {
      BOOST_STATIC_CONSTANT(bool, value =
	  (STD_TUPLE_NS::is_tuple<T>::value &&
	   STD_TUPLE_NS::tuple_size<T>::value == 1));
    };

    template <class T>
    struct tuple1_nesting_depth
      : public tuple1_nesting_depth_impl<is_tuple1<T>::value, T> {};

    template <int N, class T> struct tuple1_elt {};

    template <class T>
    struct tuple1_elt<0, T> {
      typedef typename T::value_type type;
      typedef typename boost::add_reference<type>::type ref;
      typedef typename boost::add_reference<typename boost::add_const<type>::type>::type cref;

      static ref get(T& x) {return x.value;}
      static cref cget(const T& x) {return x.value;}
    };

    template <bool Wrapit, class T>
    struct one_arg_constructor_wrapper {
      static const T& run(const T& value) {
	return value;
      }
    };

    template <class T>
    struct one_arg_constructor_wrapper<true, T> {
      static typename boost::add_reference<typename boost::add_const<typename tuple_element<0,T>::type>::type>::type
      run(const T& value) {
	return STD_TUPLE_NS::get<(0)>(value);
      }
    };

    template <class T>
    class tuple1 {
      T value;
      typedef T value_type;

      template <int N, class U> friend class tuple1_elt;

      public:
      typedef boost::type_traits::yes_type I_am_a_std_tuple_normal_tuple_class;
      tuple1(): value() {}
      tuple1(const tuple1& o): value(o.value) {}

      // Construct from member or 1-tuple
      template <class U>
      explicit tuple1(const U& v): value(
	one_arg_constructor_wrapper<
	  (tuple1_nesting_depth<U>::value > tuple1_nesting_depth<T>::value),
	  U>::run(v)) {}
      explicit tuple1(typename boost::add_reference<typename boost::add_const<T>::type>::type v): value(v) {}

      template <class U>
      tuple1(U& v, int): value(v) {}
      template <class U>
      tuple1& operator=(const U& v) {
	value = STD_TUPLE_NS::get<0>(v);
	return *this;
      }

      tuple1& operator=(const tuple1& v) {
	value = STD_TUPLE_NS::get<0>(v);
	return *this;
      }
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
