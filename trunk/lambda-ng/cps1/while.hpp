#ifndef LAMBDA_WHILE_HPP
#define LAMBDA_WHILE_HPP

#include "if.hpp"
// For ignore_cont

template <class Env, class Test, class Body, class K>
struct while_helper {
  Env env; Test test; Body body; K k;
  while_helper(const Env& env_, const Test& test_, const Body& body_, const K& k_):
    env(env_), test(test_), body(body_), k(k_) {}
  void operator()(bool b) const {
    if (b) {
      body.run(env,ignore_cont());
      while_(test,body).run(env,k);
    } else {
      k(0);
    }
  }
};

template <class Test, class Body>
struct while_type: public expr_node {
  Test test; Body body;
  while_type(const Test& test_, const Body& body_):
    test(test_), body(body_) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    test.run(env, while_helper<Env,Test,Body,K>(env,test,body,k));
  }
};

template <class Test, class Body>
while_type<typename node_traits<Test>::wrapped,
	   typename node_traits<Body>::wrapped>
while_(const Test& test, const Body& body) {
  return while_type<typename node_traits<Test>::wrapped,
		    typename node_traits<Body>::wrapped>
    (wrap(test),wrap(body));
}

#endif // LAMBDA_WHILE_HPP
