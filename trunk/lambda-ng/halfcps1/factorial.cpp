#include <iostream>
#include "boost/type_traits.hpp"
#include "tags.hpp"
#include "env.hpp"
#include "var.hpp"
#include "constant.hpp"
#include "let.hpp"
#include "print.hpp"
#include "lambda.hpp"
#if 0
#include "apply.hpp"
#include "begin.hpp"
#include "if.hpp"
#include "while.hpp"
#include "set.hpp"
#endif

struct runner {
  template <class Prog>
  void operator()(const Prog& p) const {
    p.run(empty_env);
  }
};

template <class Prog>
void run(const Prog& p) {
  p.template get_type<empty_env_type>(runner());
}

#if 0
template <template <class> class Evaluator>
struct func_type {
  typedef func_type<Evaluator> self;
  LAMBDA_NODE_CONTENTS
  struct arg_type {};
  arg_type argname() const {return arg_type();}
  empty_env_type env;
  struct body_type {
    template <class Env, class K>
    void run(const Env& env, const K& k) const {
      env.get(arg_type(), Evaluator<K>(k));
    }
  };
  body_type body;
};

#define FUNC(name, body) \
template <class K> \
struct name##_helper { \
  K k; \
  name##_helper(const K& k_): k(k_) {} \
  template <class T> \
  void operator()(const T& x) const { \
    k(body); \
  } \
}; \
func_type<name##_helper> name

template <class Evaluator, class StoredValue>
struct func_curried_type {
  typedef func_curried_type<Evaluator,StoredValue> self;
  LAMBDA_NODE_CONTENTS
  struct arg_type {};
  arg_type argname() const {return arg_type();}
  empty_env_type env;
  struct body_type {
    StoredValue value;
    body_type(const StoredValue& sv): value(sv) {}
    template <class Env, class K>
    void run(const Env& env, const K& k) const {
      env.get(arg_type(), typename Evaluator::template with_k<K>(k,value));
    }
  };
  body_type body;
  func_curried_type(const StoredValue& sv): body(sv) {}
};

#define FUNC2(name, body) \
template <class SV> \
struct name##_2_helper { \
  template <class K> \
  struct with_k { \
    K k; SV x; \
    with_k(const K& k_, const SV& sv_): k(k_), x(sv_) {} \
    template <class T> \
    void operator()(const T& y) const { \
      k(body); \
    } \
  }; \
}; \
template <class SV> \
func_curried_type<name##_2_helper<SV>, SV> \
name##_2(const SV& sv) { \
  return func_curried_type<name##_2_helper<SV>, SV>(sv); \
} \
FUNC(name, name##_2(x))

FUNC(decrement, x-1);
FUNC2(multiply, x*y);
FUNC(is_zero, (x==0));
FUNC(not_zero, (x!=0));

template <bool B, class T>
struct enable_if {
  typedef T type;
};

template <class T>
struct enable_if<false, T> {};

template <bool B, class T>
struct disable_if: public enable_if<!B, T> {};

template <class A, class B>
struct sequence_node_type: public expr_node {
  typedef sequence_node_type<A,B> self;
  LAMBDA_NODE_CONTENTS
  A a; B b;
  sequence_node_type(const A& a_, const B& b_): a(a_), b(b_) {}

  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    begin(a,b).run(env,k);
  }
};

template <class T, class U>
typename disable_if<node_traits<T>::value == is_constant
		    && node_traits<U>::value == is_constant,
		    sequence_node_type<T,U> >::type
operator,(const T& t, const U& u) {
  return sequence_node_type<T,U>(t,u);
}

// locals<int, float>(1, 3.14)(loc1 + loc2)
#endif

LAMBDA_VAR(a); LAMBDA_VAR(b); LAMBDA_VAR(fac); LAMBDA_VAR(n); LAMBDA_VAR(acc);

int main(int, char**) {
  using namespace std;
#if 0
  cout << run<int>(
      let(fac = lambda(fac,
		  lambda(acc,
		    lambda(n,
		      if_(is_zero(n),
			acc,
			fac(fac)(multiply(n, acc), decrement(n)))))),
		fac(fac)(1, 5))
      // fac is fac --> int --> int
      //let(a,3,begin(5,apply(apply(multiply,7),
      //		 apply(decrement,a))))
      // apply(lambda(a,5), lambda(a,apply(a,a)))
      ) << endl;
#endif
  run(let(a,5,print(6)));
  run(print(let(a,5,6)));
  run(let(a,5,print(a)));
  run(print(let(a,5,a)));
  run(print(apply(lambda<int>(a,5), 6)));
#if 0
  run<void>(
    let((n = 5, acc = 1),
	(while_(not_zero(n),
	  (acc = multiply(n, acc),
	   n = decrement(n))),
	 print(acc))));
#endif
  return 0;
}
