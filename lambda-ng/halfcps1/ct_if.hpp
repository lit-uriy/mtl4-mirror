#ifndef LAMBDA_CT_IF_HPP
#define LAMBDA_CT_IF_HPP

template <bool B, class T, class E>
struct ct_if {typedef T type;};

template <class T, class E>
struct ct_if<false,T,E> {typedef E type;};

#endif // LAMBDA_CT_IF_HPP
