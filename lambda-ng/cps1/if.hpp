#ifndef LAMBDA_IF_HPP
#define LAMBDA_IF_HPP

struct ignore_cont {
  template <class T>
  void operator()(const T&) const {}
};

template <class Env, class Th, class El, class K>
struct if_helper {
  Env env; Th th; El el; K k;
  if_helper(const Env& env_, const Th& th_, const El& el_, const K& k_):
    env(env_), th(th_), el(el_), k(k_) {}
  void operator()(bool b) const {
    if (b)
      th.run(env,ignore_cont());
    else
      el.run(env,ignore_cont());
    k(0);
  }
};

template <class Test, class Th, class El>
struct if_type: public expr_node {
  Test test; Th th; El el;
  if_type(const Test& test_, const Th& th_, const El& el_):
    test(test_), th(th_), el(el_) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    test.run(env, if_helper<Env,Th,El,K>(env,th,el,k));
  }
};

template <class Test, class Th, class El>
if_type<typename node_traits<Test>::wrapped,
	typename node_traits<Th>::wrapped,
	typename node_traits<El>::wrapped>
if_(const Test& test, const Th& th, const El& el) {
  return if_type<typename node_traits<Test>::wrapped,
                 typename node_traits<Th>::wrapped,
                 typename node_traits<El>::wrapped>
    (wrap(test),wrap(th),wrap(el));
}

#endif // LAMBDA_IF_HPP
