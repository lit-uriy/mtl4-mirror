#include <iostream>
#include "boost/type_traits.hpp"

#define FUNCTOR0(name) struct name {};

#define FUNCTOR2(name) \
template <class A, class B> \
struct name##_type { \
  A a; B b; \
  name##_type(const A& x, const B& y): a(x), b(y) {} \
}; \
template <class A, class B> \
name##_type<A,B> name(const A& a, const B& b) { \
  return name##_type<A,B>(a,b); \
}

template <class Env, class Var, class Value>
struct extended_env;

struct empty_env_type {
  //  template <class Var, class K>
  // void get(const Var&, const K& k) const;
  template <class K>
  void get(const K&) const;

  template <class Var, class Value>
  extended_env<empty_env_type, Var, Value>
  extend(const Var&, const Value& v) const {
    return extended_env<empty_env_type, Var, Value>(*this, v);
  }
};

empty_env_type empty_env;

template <class Env, class Var, class Value>
struct extended_env: public Env {
  Value v;
  extended_env(const Env& e_, const Value& v_): Env(e_), v(v_) {}

  template <class T, class K>
  void get(const T& t, const K& k) const {
    Env::get(t,k);
  }

  template <class K>
  void get(const Var&, const K& k) const {
    k(v);
  }

  template <class Var2, class Value2>
  extended_env<extended_env<Env,Var,Value>, Var2, Value2>
  extend(const Var2&, const Value2& v) const {
    return extended_env<extended_env<Env,Var,Value>, Var2, Value2>(*this, v);
  }
};

FUNCTOR0(nil)
FUNCTOR2(cons)

struct expr_node {}; struct var_node {};

enum node_type {is_constant, is_var, is_expr};

template <class T, node_type NT> struct wrap_node_traits {};

template <class T> struct wrap_node_traits<T, is_expr> {
  typedef T type;
};

template <class T>
struct node_traits {
  static const node_type value =
    boost::is_base_and_derived<expr_node, T>::value ? is_expr :
    boost::is_base_and_derived<var_node, T>::value ? is_var : is_constant;
  typedef typename wrap_node_traits<T,value>::type wrapped;
};

template <class T>
typename node_traits<T>::wrapped wrap(const T& x) {
  return (typename node_traits<T>::wrapped)x;
}

template <class C>
struct constant_type: public expr_node {
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

template <class T> struct wrap_node_traits<T,is_constant> {
  typedef constant_type<T> type;
};

template <class V>
struct var_type: public expr_node {
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

template <class T> struct wrap_node_traits<T,is_var> {
  typedef var_type<T> type;
};

template <class Var, class Env, class Body, class K>
struct let_helper {
  Env env; Body body; K k;
  let_helper(const Env& env_, const Body& body_, const K& k_):
    env(env_), body(body_), k(k_) {}
  template <class Value>
  void operator()(const Value& value) const {
    body.run(env.extend(Var(), value), k);
  }
};

template <class Var, class Value, class Body>
struct let_type: public expr_node {
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

template <class Proc, class Env, class K>
struct apply_helper_2 {
  Proc proc; Env env; K k;
  apply_helper_2(const Proc& proc_, const Env& env_, const K& k_):
    proc(proc_), env(env_), k(k_) {}

  template <class Arg>
  void operator()(const Arg& arg) const {
    proc.body.run(proc.env.extend(proc.argname(), arg), k);
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

template <class T>
struct writer_type {
  mutable T& x;
  writer_type(T& x_): x(x_) {}
  void operator()(const T& v) const {x = v;}
};
template <class T>
writer_type<T> writer(T& x) {return x;}

template <class Ret, class Prog>
Ret run(const Prog& p) {
  Ret result = -999;
  p.run(empty_env, writer(result));
  return result;
}

#define VAR(v) struct v##_: public var_node {}; v##_ v

template <class Body1, class Body2>
struct begin_type: public expr_node {
  VAR(a);
  Body1 body1; Body2 body2;

  begin_type(const Body1& body1_, const Body2& body2_):
    body1(body1_), body2(body2_) {}
  
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    let(a,body1,body2).run(env,k);
  }
};
template <class Body1, class Body2>
begin_type<typename node_traits<Body1>::wrapped,typename node_traits<Body2>::wrapped>
begin(const Body1& body1, const Body2& body2) {
  return begin_type<typename node_traits<Body1>::wrapped,typename node_traits<Body2>::wrapped>(wrap(body1), wrap(body2));
}

template <class K>
struct print_helper {
  K k;
  print_helper(const K& k_): k(k_) {}

  template <class T>
  void operator()(const T& x) const {
    std::cout << x << std::endl;
    k(0);
  }
};

template <class Expr>
struct print_type: public expr_node {
  Expr expr;
  print_type(const Expr& expr_): expr(expr_) {}
  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    expr.run(env, print_helper<K>(k));
  }
};
template <class Expr>
print_type<typename node_traits<Expr>::wrapped> print(const Expr& expr) {
  return print_type<typename node_traits<Expr>::wrapped>(wrap(expr));
}

template <template <class> class Evaluator>
struct func_type {
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
    std::cout << "Running " #name " on " << x << std::endl; \
    k(body); \
  } \
}; \
func_type<name##_helper> name

template <class Evaluator, class StoredValue>
struct func_curried_type {
  struct arg_type {};
  arg_type argname() const {return arg_type();}
  empty_env_type env;
  struct body_type {
    StoredValue value;
    body_type(const StoredValue& sv): value(sv) {}
    template <class Env, class K>
    void run(const Env& env, const K& k) const {
      std::cout << "Running curried" << std::endl;
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
      std::cout << "x=" << x << " y=" << y << std::endl; \
      k(body); \
    } \
  }; \
}; \
template <class SV> \
func_curried_type<name##_2_helper<SV>, SV> \
name##_2(const SV& sv) { \
  std::cout << "running " #name "_2" << std::endl; \
  return func_curried_type<name##_2_helper<SV>, SV>(sv); \
} \
FUNC(name, name##_2(x))

template <class Env, class Th, class El, class K>
struct if_helper {
  Env env; Th th; El el; K k;
  if_helper(const Env& env_, const Th& th_, const El& el_, const K& k_):
    env(env_), th(th_), el(el_), k(k_) {}
  void operator()(bool b) const {
    if (b)
      th.run(env,k);
    else
      el.run(env,k);
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

FUNC(decrement, x-1);
FUNC2(multiply, x*y);
FUNC(is_zero, (x==0));
VAR(a); VAR(b); VAR(fac); VAR(n); VAR(acc);

// locals<int, float>(1, 3.14)(loc1 + loc2)

int main(int, char**) {
  using namespace std;
  cout << run<int>(
		   let(fac,
		       lambda(fac,
			      lambda(acc,
				     lambda(n,
					    if_(apply(is_zero,n),
						acc,
						apply(
						 apply(apply(fac,fac),
						       apply(apply(multiply,n),acc)),
						 apply(decrement,n)))))),
		       apply(apply(apply(fac,fac),1),5))
		   // fac is fac --> int --> int
		   //let(a,3,begin(5,apply(apply(multiply,7),
		   //		 apply(decrement,a))))
		   // apply(lambda(a,5), lambda(a,apply(a,a)))
    ) << endl;
  return 0;
}
