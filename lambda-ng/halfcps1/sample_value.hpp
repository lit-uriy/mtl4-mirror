#ifndef LAMBDA_SAMPLE_VALUE_HPP
#define LAMBDA_SAMPLE_VALUE_HPP

#include "boost/type_traits.hpp"

template <class T> struct unbound_variable {
  private: unbound_variable();
};

template <class T>
typename boost::remove_reference<T>::type
sample_value() {
  return typename boost::remove_reference<T>::type();
}

#endif // LAMBDA_SAMPLE_VALUE_HPP
