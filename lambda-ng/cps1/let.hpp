#ifndef LAMBDA_LET_HPP
#define LAMBDA_LET_HPP

#include "tags.hpp"
#include "assign.hpp"

template <class Var, class Env, class Body, class K>
struct let_helper {
  Env env; Body body; K k;
  let_helper(const Env& env_, const Body& body_, const K& k_):
    env(env_), body(body_), k(k_) {}
  template <class Value>
  void operator()(const Value& value) const {
    Value v = value; // Make copy so it is not const anymore
    body.run(env.extend(Var(), v), k);
  }
};

template <class Var, class Value, class Body>
struct let_type: public expr_node {
  typedef let_type<Var,Value,Body> self;
  // LAMBDA_NODE_CONTENTS
  Value value; Body body;
  let_type(const Value& value_, const Body& body_): value(value_), body(body_) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    value.run(env, let_helper<Var,Env,Body,K>(env, body, k));
  }
};
template <class Var, class Value, class Body>
let_type<Var,typename node_traits<Value>::wrapped,typename node_traits<Body>::wrapped>
let(const Var&, const Value& value, const Body& body) {
  return let_type<Var,typename node_traits<Value>::wrapped,typename node_traits<Body>::wrapped>(wrap(value),wrap(body));
}

template <class Vars, class Body> struct let_traits {};

template <class Var, class Value, class Body>
struct let_traits<assign_node_type<Var,Value>, Body> {
  typedef let_type<typename Var::underlying_var_type,typename node_traits<Value>::wrapped,typename node_traits<Body>::wrapped> type;
};

template <class A, class B> struct sequence_node_type;

template <class A, class B, class Body>
struct let_traits<sequence_node_type<A,B>, Body> {
  typedef typename let_traits<A,
	    typename let_traits<B, Body>::type>::type type;
};

template <class Var, class Value, class Body>
typename let_traits<assign_node_type<Var,Value>, Body>::type
let(const assign_node_type<Var,Value>& vv, const Body& body) {
  return typename let_traits<assign_node_type<Var,Value>, Body>::type(wrap(vv.rhs),wrap(body));
}

template <class A, class B, class Body>
typename let_traits<sequence_node_type<A,B>, Body>::type
let(const sequence_node_type<A,B>& vv, const Body& body) {
  return let(vv.a, let(vv.b, body));
}

#endif // LAMBDA_LET_HPP
