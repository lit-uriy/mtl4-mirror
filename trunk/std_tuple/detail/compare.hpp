#ifndef STD_TUPLE_COMPARE_HPP
#define STD_TUPLE_COMPARE_HPP

#ifndef STD_TUPLE_IN_LIB
#error "Must include <tuple> for this library"
#endif

#include "traits.hpp"

namespace STD_TUPLE_NS { // To allow for change to std later
  namespace detail {
    // Tuple functions
    template <int Start, int Length, class T1, class T2>
    struct tuple_short_circuit {
      const T1& t1; const T2& t2;
      tuple_short_circuit(const T1& t1_, const T2& t2_): t1(t1_), t2(t2_) {}
      typedef tuple_short_circuit<Start+1, Length, T1, T2> next;
      bool eq() const {return get<Start>(t1) == get<Start>(t2) &&
			      next(t1,t2).eq();}
      bool ne() const {return get<Start>(t1) != get<Start>(t2) ||
			      next(t1,t2).ne();}
      bool lt() const {
	if (get<Start>(t1) < get<Start>(t2)) return true;
	if (get<Start>(t2) < get<Start>(t1)) return false;
	return next(t1,t2).lt();
      }
      bool gt() const {
	if (get<Start>(t1) > get<Start>(t2)) return true;
	if (get<Start>(t2) > get<Start>(t1)) return false;
	return next(t1,t2).gt();
      }
      bool le() const {
	if (!(bool)(get<Start>(t1) <= get<Start>(t2))) return false;
	if (!(bool)(get<Start>(t2) <= get<Start>(t1))) return true;
	return next(t1,t2).le();
      }
      bool ge() const {
	if (!(bool)(get<Start>(t1) >= get<Start>(t2))) return false;
	if (!(bool)(get<Start>(t2) >= get<Start>(t1))) return true;
	return next(t1,t2).ge();
      }
    };

    template <int Start, class T1, class T2>
    struct tuple_short_circuit<Start,Start,T1,T2> {
      tuple_short_circuit(const T1&, const T2&) {}
      bool eq() const {return true;}
      bool ne() const {return false;}
      bool lt() const {return false;}
      bool gt() const {return false;}
      bool le() const {return true;}
      bool ge() const {return true;}
    };
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_eq(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).eq();
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_ne(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).ne();
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_lt(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).lt();
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_gt(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).gt();
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_le(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).le();
  }

  template <class T1, class T2>
  typename detail::enable_if<tuple_size<T1>::value == tuple_size<T2>::value,
			     bool>::type
  tuple_ge(const T1& t1, const T2& t2) {
    return detail::tuple_short_circuit<0,tuple_size<T1>::value,T1,T2>(t1,t2).ge();
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator ==(const T1& t1, const T2& t2) {
    return tuple_eq(t1,t2);
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator !=(const T1& t1, const T2& t2) {
    return tuple_ne(t1,t2);
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator <(const T1& t1, const T2& t2) {
    return tuple_lt(t1,t2);
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator >(const T1& t1, const T2& t2) {
    return tuple_gt(t1,t2);
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator <=(const T1& t1, const T2& t2) {
    return tuple_le(t1,t2);
  }

  template <class T1, class T2>
  typename detail::enable_if<is_tuple<T1>::value && is_tuple<T2>::value,
			     bool>::type
  operator >=(const T1& t1, const T2& t2) {
    return tuple_ge(t1,t2);
  }
}

#endif // STD_TUPLE_COMPARE_HPP
