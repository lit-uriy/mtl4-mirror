#ifndef LAMBDA_ENV_HPP
#define LAMBDA_ENV_HPP

template <class Env, class Var, class Value>
struct extended_env;

struct empty_env_type {
  template <class K>
  void get(const K&) const;

  template <class Var, class Value>
  extended_env<empty_env_type, Var, Value>
  extend(const Var&, Value& v) const {
    return extended_env<empty_env_type, Var, Value>(*this, v);
  }
};

empty_env_type empty_env;

template <class Env, class Var, class Value>
struct extended_env: public Env {
  Value& v;
  extended_env(const Env& e_, Value& v_): Env(e_), v(v_) {}

  template <class T, class K>
  void get(const T& t, const K& k) {
    Env::get(t,k);
  }

  template <class K>
  void get(const Var&, const K& k) {
    k(v);
  }

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
  extend(const Var2&, Value2& v) const {
    return extended_env<extended_env<Env,Var,Value>, Var2, Value2>(*this, v);
  }
};

#endif // LAMBDA_ENV_HPP
