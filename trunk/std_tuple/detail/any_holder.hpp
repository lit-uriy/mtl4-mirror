#ifndef STD_TUPLE_ANY_HOLDER_HPP
#define STD_TUPLE_ANY_HOLDER_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"
#include "boost/type_traits.hpp"

// Any_holder and swallow_assign from tuple standard, plus related functions

namespace STD_TUPLE_NS {
  template <class T>
  class any_holder {
    T data;

    public:
    typedef T type;

    operator T() {return data;}
    T unwrap() {return data;}

    any_holder(typename boost::add_reference<
		 typename boost::add_const<T>::type
	       >::type t): data(t) {}
  };

  template <class T>
  inline any_holder<T&> ref(T& t) {
    return any_holder<T&>(t);
  }

  template <class T>
  inline any_holder<const T&> cref(const T& t) {
    return any_holder<const T&>(t);
  }


  struct swallow_assign {
    template <class T>
    swallow_assign& operator=(const T&) {return *this;}
  };

  static swallow_assign ignore;
}

#endif // STD_TUPLE_ANY_HOLDER_HPP
