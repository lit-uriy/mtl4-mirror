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
#include <boost/numeric/mtl/interface/vpt.hpp>

using namespace std;  

void helper_function()
{
    mtl::vpt::vampir<199> tracer; 

    std::cout << "In helper_function <id=199>, it is " << (tracer.is_traced() ? "" : "not ") << "traced\n";
}

void function()
{
    mtl::vpt::vampir<299> tracer; 

    std::cout << "In function <id=299>, it is " << (tracer.is_traced() ? "" : "not ") << "traced\n";
    helper_function();
}



int test_main(int, char**) 
{
    mtl::vpt::vampir<999> tracer;

    helper_function();
    function();

    return 0; 
}

