#ifndef LAMBDA_SET_HPP
#define LAMBDA_SET_HPP

#include "tags.hpp"

template <class Value, class Env, class K>
struct set_helper_2 {
  Env env; K k; Value value;
  set_helper_2(const Env& env_, const K& k_, const Value& value_):
    env(env_), k(k_), value(value_) {}
  template <class Ref>
  void operator()(Ref& ref) const {
    ref = value;
    k(value);
  }
};

template <class Var, class Env, class K>
struct set_helper {
  Env env; K k;
  set_helper(const Env& env_, const K& k_):
    env(env_), k(k_) {}
  template <class Value>
  void operator()(const Value& value) const {
    env.get(Var(), set_helper_2<Value,Env,K>(env,k,value));
  }
};

template <class Var, class Value>
struct set_type: public expr_node {
  typedef set_type<Var,Value> self;
  LAMBDA_NODE_CONTENTS
  Value value;
  set_type(const Value& value_): value(value_) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    value.run(env, set_helper<Var,Env,K>(env, k));
  }
};
template <class Var, class Value>
set_type<Var,typename node_traits<Value>::wrapped>
set(const Var&, const Value& value) {
  return set_type<Var,typename node_traits<Value>::wrapped>(wrap(value));
}

template <class Var, class Value>
set_type<Var,typename node_traits<Value>::wrapped>
set(const var_type<Var>&, const Value& value) {
  return set_type<Var,typename node_traits<Value>::wrapped>(wrap(value));
}

#endif // LAMBDA_SET_HPP
