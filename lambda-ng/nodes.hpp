#ifndef LAMBDA_NODES_HPP
#define LAMBDA_NODES_HPP

struct expr_node {};
struct var_node {};
struct constant_node {};

template <class T>
struct constant_type {
  T value;
  constant_type(): value() {}
  constant_type(const T& x): value(x) {}
  template <class Env>
  T run(const Env&) const {
    return value;
  }
};

template <class Var>
struct variable_type {
  Var v;
  variable_type(): v() {}
  variable_type(const Var& v_): v(v_) {}
  template <class Env>
  typeof(Env().get(v)) run(const Env& env) const {
    return env.get(v);
  }
};

template <class T>
T wrap_(const T& x, const expr_node* const) {return x;}

template <class T>
constant_type<T> wrap_(const T& x, const void* const) {return x;}

template <class Var>
variable_type<Var> wrap_(const Var& x, const var_node* const) {return x;}

template <class T>
typeof(wrap_(T(), (T*)(0)))
wrap(const T& x) {return wrap_(x,&x);}

#define VAR(name) struct name##_: public var_node {}; name##_ name

#endif // LAMBDA_NODES_HPP
