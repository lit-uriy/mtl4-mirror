// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschrÃ¤nkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;

template <typename T>
struct lazy_t
{
    lazy_t(T& data) : data(data) {}
    T& data;
};

template <typename T>
inline lazy_t<T> lazy(T& x) 
{ return lazy_t<T>(x); }

template <typename T>
inline lazy_t<const T> lazy(const T& x) 
{ return lazy_t<const T>(x); }



int main(int, char**) 
{
    

    return 0;
}
