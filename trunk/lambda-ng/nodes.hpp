#ifndef LAMBDA_NODES_HPP
#define LAMBDA_NODES_HPP

struct expr_node {};
struct var_node {};
struct constant_node {};

template <class T>
struct constant_type;

template <class Var>
struct variable_type;

#define LAMBDA_AUTO_FUNC(proto, body) typeof(body) proto {return (body);}
#define LAMBDA_AUTO(name, value) typeof((value)) name = (value)

template <class T>
T wrap_(const T& x, expr_node) {return x;}

template <class T>
constant_type<T> wrap_(const T& x, constant_node) {return x;}

template <class Var>
var_type<Var> wrap_(const T& x, var_node) {return var_type<Var>();}

template <class T>
LAMBDA_AUTO_FUNC(wrap(const T& x), wrap_(x,x))

#define VAR(name) struct name##_: public var_node {}; name##_ name

#endif // LAMBDA_NODES_HPP
