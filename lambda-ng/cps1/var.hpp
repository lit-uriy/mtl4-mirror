#ifndef LAMBDA_VAR_HPP
#define LAMBDA_VAR_HPP

#include "tags.hpp"

template <class V>
struct var_type: public expr_node {
  typedef V underlying_var_type;
  typedef var_type<V> self;
  LAMBDA_NODE_CONTENTS
  var_type(const V&) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    env.get(V(),k);
  }
};
template <class V>
var_type<V> var(const V&) {
  return var_type<V>();
}

#define LAMBDA_VAR(v) struct v##_: public var_node { \
			typedef v##_ self; \
			LAMBDA_NODE_CONTENTS \
		      }; \
		      v##_ v

#endif // LAMBDA_VAR_HPP
