#ifndef STD_TUPLE_TRAITS_HPP
#define STD_TUPLE_TRAITS_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "boost/type_traits.hpp"
#include "utility.hpp"

namespace STD_TUPLE_NS {
  namespace detail {
    struct tuple_tag {};
  }

  //User-visible tuple operations
  template <class T>
  struct is_tuple {
    BOOST_STATIC_CONSTANT(bool, value = (boost::is_base_and_derived<detail::tuple_tag, T>::value));
  };

  template <class T>
  struct is_tuple_like: public is_tuple<T> {};

  namespace detail {
    struct tuple0_tag: public tuple_tag {};
    struct tuple1_tag: public tuple_tag {};
    struct append_tuple_tag: public tuple_tag {};
    struct subrange_tag: public tuple_tag {};

    enum {
      nontuple = 100, user_tuple, 
      tuple0_derived, tuple1_derived, append_tuple_derived,
      subrange_derived
    };

    template <int Tc, class T> struct tuple_traits_impl {};

    template <class T>
    struct is_user_tuple {
      BOOST_STATIC_CONSTANT(bool, value = (is_tuple_like<T>::value && !is_tuple<T>::value));
    };

    void categorize_tuple(); // All real versions will have one arg

    template <class T>
    struct tuple_category {
      BOOST_STATIC_CONSTANT(int, value =
	(is_user_tuple<T>::value
	  ? user_tuple
	  : sizeof(*STD_TUPLE_NS::detail::categorize_tuple((typename boost::remove_const<typename boost::remove_reference<T>::type>::type *)(0)))));
    };

    template <class T>
    struct tuple_traits
      : public tuple_traits_impl<tuple_category<T>::value, T>
      {};

    // Type-integer encoding/decoding based on "A Portable typeof Operator"
    // by Bill Gibbons in the May 2000 issue of the ACCU newsletter.

    template <int N>
    struct encode_type {
      typedef char (*type)[N];
    };
  }

  template <class T>
  struct tuple_size {
    BOOST_STATIC_CONSTANT(int, value = detail::tuple_traits<T>::size);
  };

  template <int N, class T>
  struct tuple_element {
    typedef typename detail::tuple_traits<T>::template element<N>::type type;
  };

  template <int N, class T>
  typename detail::disable_if<boost::is_const<T>::value,
    typename detail::tuple_traits<T>::template element<N>::ref
  >::type
  get(T& x) {
    typedef typename detail::tuple_traits<T>::template element<N> tt;
    return tt::get(x);
  }

  template <int N, class T>
  typename detail::tuple_traits<T>::template element<N>::cref
  get(const T& x) {
    typedef typename detail::tuple_traits<T>::template element<N> tt;
    return tt::cget(x);
  }
}

#endif // STD_TUPLE_TRAITS_HPP
