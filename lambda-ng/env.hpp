#ifndef LAMBDA_ENV_HPP
#define LAMBDA_ENV_HPP

template <class Env, class Var, class Value>
struct extended_env;

struct empty_env_type {
  struct unbound_var_type;
  void get(const unbound_var_type&);

  template <class Value, class Var>
  extended_env<empty_env_type, Var, Value>
  extend(const Var&, const Value& value = Value()) {
    return extended_env<empty_env_type, Var, Value>(*this, value);
  }
};

empty_env_type empty_env;

template <class Env, class Var, class Value>
struct extended_env: public Env {
  Value& value;

  extended_env(Value& v): value(v) {}

  Value& get(const Var&) {
    return value;
  }

  const Value& get(const Var&) const {
    return value;
  }

  template <class Var2>
  typeof(Env::get(Var2())) get(const Var2&) {
    return Env::get(Var2());
  }

  template <class Var2>
  typeof(Env::get(Var2())) get(const Var2&) const {
    return Env::get(Var2());
  }

  template <class Value2, class Var2>
  extended_env<extended_env<Env,Var,Value>, Var2, Value2>
  extend(const Var2&, Value2& value) {
    return extended_env<extended_env<Env,Var,Value>, Var2, Value2>(*this, value);
  }
};

#endif // LAMBDA_ENV_HPP
