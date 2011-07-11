// Software License for PMTL
// 
// Copyright (c) 2010 SimuNova UG, www.simunova.com.
// All rights reserved.
// Author: Peter Gottschling
// 
// This file is part of the Parallel Matrix Template Library
// 
// The details are regulated by the EULA at http://www.simunova.com/en/eula
//                             respectively http://www.simunova.com/de/agb.

#define MTL_VPT_LEVEL 2

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace std;  

void helper_function()
{
    mtl::vampir_trace<1099> tracer; 

    std::cout << "In helper_function <id=1099>, it is " << (tracer.is_traced() ? "" : "not ") << "traced\n";
}

void function()
{
    mtl::vampir_trace<2099> tracer; 

    std::cout << "In function <id=2099>, it is " << (tracer.is_traced() ? "" : "not ") << "traced\n";
    helper_function();
}



int test_main(int, char**) 
{
    mtl::vampir_trace<9999> tracer;

    helper_function();
    ::function(); // explicit nomination because VS2010 has ambiguity with std::tr1::function

    return 0; 
}

