#ifndef LAMBDA_ENV_HPP
#define LAMBDA_ENV_HPP

#include "boost/type_traits.hpp"
#include "ct_if.hpp"

template <class Env, class Var, class Value>
struct extended_env;

struct empty_env_type {
  template <class T>
  void get(const T&) const;

  template <class T>
  struct get_var_type {};

  template <class Var, class Value>
  extended_env<empty_env_type, Var, Value>
  extend(const Var&, Value& v) const {
    return extended_env<empty_env_type, Var, Value>(*this, v);
  }

  template <class Var, class Value>
  struct extend_abs {
    typedef extended_env<empty_env_type, Var, Value> type;
  };
};

empty_env_type empty_env;

template <class Env, class Var, class Value>
struct extended_env: public Env {
  Value& v;
  extended_env(const Env& e_, Value& v_): Env(e_), v(v_) {}

  template <class T>
  typename Env::template get_var_type<T>::type
  get(const T& t) const {
    return Env::get(t);
  }

  Value& get(const Var&) const {
    return v;
  }

  template <class T>
  struct get_var_type {
    typedef typename ct_if<boost::is_same<T,Var>::value,
			   value_holder<Value&>,
			   typename Env::template get_var_type<T> >::type::type
	    type;
  };

  template <class Var2, class Value2>
  extended_env<extended_env<Env,Var,Value>, Var2, Value2>
  extend(const Var2&, Value2& v) const {
    return extended_env<extended_env<Env,Var,Value>, Var2, Value2>(*this, v);
  }

  template <class Var2, class Value2>
  struct extend_abs {
    typedef extended_env<extended_env<Env,Var,Value>, Var2, Value2> type;
  };
};

#endif // LAMBDA_ENV_HPP
