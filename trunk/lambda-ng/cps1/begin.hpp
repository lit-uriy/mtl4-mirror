#ifndef LAMBDA_BEGIN_HPP
#define LAMBDA_BEGIN_HPP

#include "if.hpp"

template <class Body1, class Body2>
struct begin_type: public expr_node {
  typedef begin_type<Body1,Body2> self;
  LAMBDA_NODE_CONTENTS
  LAMBDA_VAR(a);
  Body1 body1; Body2 body2;

  begin_type(const Body1& body1_, const Body2& body2_):
    body1(body1_), body2(body2_) {}
  
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    body1.run(env,ignore_cont());
    body2.run(env,k);
  }
};
template <class Body1, class Body2>
begin_type<typename node_traits<Body1>::wrapped,typename node_traits<Body2>::wrapped>
begin(const Body1& body1, const Body2& body2) {
  return begin_type<typename node_traits<Body1>::wrapped,typename node_traits<Body2>::wrapped>(wrap(body1), wrap(body2));
}

#endif // LAMBDA_BEGIN_HPP
