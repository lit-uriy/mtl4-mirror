#ifndef LAMBDA_APPLY_HPP
#define LAMBDA_APPLY_HPP

#include "tags.hpp"

template <class Proc, class Env, class K>
struct apply_helper_2 {
  Proc proc; Env env; K k;
  apply_helper_2(const Proc& proc_, const Env& env_, const K& k_):
    proc(proc_), env(env_), k(k_) {}

  template <class Arg>
  void operator()(const Arg& arg) const {
    Arg a = arg; // Force copy
    proc.body.run(proc.env.extend(proc.argname(), a), k);
  }
};

template <class Rand, class Env, class K>
struct apply_helper_1 {
  Rand rand; Env env; K k;
  apply_helper_1(const Rand& rand_, const Env& env_, const K& k_):
    rand(rand_), env(env_), k(k_) {}

  template <class Proc>
  void operator()(const Proc& proc) const {
    rand.run(env, apply_helper_2<Proc,Env,K>(proc, env, k));
  }
};

template <class Rator, class Rand>
struct apply_type: public expr_node {
  typedef apply_type<Rator,Rand> self;
  LAMBDA_NODE_CONTENTS
  Rator rator; Rand rand;
  apply_type(const Rator& rator_, const Rand& rand_): rator(rator_), rand(rand_) {}

  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    rator.run(env, apply_helper_1<Rand,Env,K>(rand,env,k));
  }
};
template <class Rator, class Rand>
apply_type<typename node_traits<Rator>::wrapped,typename node_traits<Rand>::wrapped>
apply(const Rator& rator, const Rand& rand) {
  return apply_type<typename node_traits<Rator>::wrapped,typename node_traits<Rand>::wrapped>(wrap(rator),wrap(rand));
}

#endif // LAMBDA_APPLY_HPP
