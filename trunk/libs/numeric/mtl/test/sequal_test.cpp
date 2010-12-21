// Software License for MTL
//
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
//
// This file is part of the Matrix Template Library
//
// See also license.mtl.txt in the distribution.

// With contributions from Cornelius Steinhardt

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/operation/sequal.hpp>
#include <boost/numeric/mtl/operation/sort.hpp>

using namespace std;


int test_main(int , char**)
{
    using namespace mtl;
    typedef dense_vector<double>  Vector;
    Vector                    lambda(2, 0.0), z(2), d(2);
    z[0]=1; z[1]=1;
    d[0]=-5; d[1]=-1;
   
    //lambda= mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(3.0);

    std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(3.0) <<"\n";
	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(0.0) <<"\n";
	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(-3.0) <<"\n";
	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(13.0) <<"\n";
	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).funk(113.0) <<"\n";

	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).grad_f(13.0) <<"\n";
	std::cout<<"lambda  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).grad_f(113.0) <<"\n";
	std::cout<<"roots  ="<< mtl::vector::sequal<Vector>(lambda, z, d, 5.0).roots() <<"\n";
	//std::cout<<"lambda  ="<< lambda <<"\n";

	Vector x(5, 0.0);
	for(int i = 0; i < 5; i++)
		x[i]=5-i;
	x[1]=1;
	std::cout<< "\n x=" << x << "\n";
	mtl::vector::sort<Vector>(x);
	std::cout<< "x=" << x << "\n";
	if(x[0] != 1.0) throw "Error in sorting.";

	return 0;
}



