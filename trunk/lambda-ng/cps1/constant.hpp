#ifndef LAMBDA_CONSTANT_HPP
#define LAMBDA_CONSTANT_HPP

#include "tags.hpp"

template <class C>
struct constant_type: public expr_node {
  typedef constant_type<C> self;
  LAMBDA_NODE_CONTENTS
  C value;
  constant_type(const C& c): value(c) {}

  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    k(value);
  }
};
template <class C>
constant_type<C> constant(const C& c) {
  return constant_type<C>(c);
}

#endif // LAMBDA_CONSTANT_HPP
