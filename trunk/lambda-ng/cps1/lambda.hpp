#ifndef LAMBDA_LAMBDA_HPP
#define LAMBDA_LAMBDA_HPP

template <class Var, class Body, class Env>
struct closure_type: public expr_node {
  Body body; Env env;
  Var argname() const {return Var();}
  closure_type(const Body& body_, const Env& env_): body(body_), env(env_) {}
};
template <class Var, class Body, class Env>
closure_type<Var,Body,Env>
closure(const Var&, const Body& body, const Env& env) {
  return closure_type<Var,Body,Env>(body,env);
}

template <class Var, class Body>
struct lambda_type: public expr_node {
  typedef lambda_type<Var,Body> self;
  LAMBDA_NODE_CONTENTS
  Body body;
  lambda_type(const Body& body_): body(body_) {}

  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    k(closure(Var(), body, env));
  }
};
template <class Var, class Body>
lambda_type<Var,typename node_traits<Body>::wrapped> lambda(const Var&, const Body& body) {
  return lambda_type<Var,typename node_traits<Body>::wrapped>(wrap(body));
}

#endif // LAMBDA_LAMBDA_HPP
