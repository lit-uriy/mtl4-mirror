#ifndef LAMBDA_TAGS_HPP
#define LAMBDA_TAGS_HPP

struct expr_node {}; struct var_node {};

enum node_type {is_constant, is_var, is_expr};

template <class T, node_type NT> struct wrap_node_traits {};

template <class T> struct wrap_node_traits<T, is_expr> {
  typedef T type;
};

template <class T>
struct node_traits {
  static const node_type value =
    boost::is_base_and_derived<expr_node, T>::value ? is_expr :
    boost::is_base_and_derived<var_node, T>::value ? is_var : is_constant;
  typedef typename wrap_node_traits<T,value>::type wrapped;
};

template <class T>
typename node_traits<T>::wrapped wrap(const T& x) {
  return (typename node_traits<T>::wrapped)x;
}

template <class T> struct constant_type;

template <class T> struct wrap_node_traits<T,is_constant> {
  typedef constant_type<T> type;
};

template <class T> struct var_type;

template <class T> struct wrap_node_traits<T,is_var> {
  typedef var_type<T> type;
};

template <class Rator, class Rand> struct apply_type;

template <class Rator, class Rand>
apply_type<typename node_traits<Rator>::wrapped,typename node_traits<Rand>::wrapped>
apply(const Rator& rator, const Rand& rand);

template <class Lhs, class Rhs>
struct assign_node_type;

template <class Lhs, class Rhs>
assign_node_type<typename node_traits<Lhs>::wrapped,
		 typename node_traits<Rhs>::wrapped>
assign_node(const Lhs&, const Rhs&);

template <class T, class U>
struct pick1st {typedef T type;};

#define LAMBDA_NODE_CONTENTS \
  template <class A1> \
  apply_type<typename node_traits<typename pick1st<self, A1>::type>::wrapped, \
	     typename node_traits<A1>::wrapped> \
  operator()(const A1& a1) const { \
    return apply(wrap(*this), wrap(a1)); \
  } \
  \
  template <class A1, class A2> \
  apply_type<apply_type<typename node_traits<typename pick1st<self, A1>::type>::wrapped, \
			typename node_traits<A1>::wrapped>, \
	     typename node_traits<A2>::wrapped> \
  operator()(const A1& a1, const A2& a2) const { \
    return apply(apply(wrap(*this), wrap(a1)), wrap(a2)); \
  } \
  \
  template <class T> \
  assign_node_type<typename node_traits<typename pick1st<self, T>::type>::wrapped, \
		   typename node_traits<T>::wrapped> \
  operator=(const T& x) const { \
    return assign_node(wrap(*this), wrap(x)); \
  }

#endif // LAMBDA_TAGS_HPP
