// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>


using namespace std;

namespace A {
	template <typename T> class C1 {};
}	

namespace B {
	class C2
	{
#if !defined(_MSC_VER) || _MSC_VER != 1400
		template <typename T> friend class A::C1;
#endif
	};	
}

int test_main(int argc, char* argv[])
{
    return 0;
}
