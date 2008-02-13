// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <adobe/move.hpp>


// Everything in the test is double
// How to test sparse generically? 


using namespace std;
using namespace mtl;
using detail::contiguous_memory_block; //using detail::generic_array;


// Return a matrix with move semantics
// Return also the address of the first entry to be sure that it is really moved
template <typename Block>
Block f(const Block&, double*& a00)
{
    Block b(3);
    b.data[0]= 5.0;
    a00= &b.data[0];
    return b;
}

// For blocks on heap, different addresses means that moving failed
bool compare(const contiguous_memory_block<double, false, 0>& block, double* p)
{
    return &block.data[0] != p;
}

// For blocks on stack, equal addresses means accidental moving 
bool compare(const contiguous_memory_block<double, true, 3>& block, double* p)
{
    return &block.data[0] == p;
}



#if 0
bool compare(const generic_array<double, false, 0>& block, double* p)
{
    return &block.data[0] != p;
}

bool compare(const mtl::detail::generic_array<double, true, 3u>& block, double*& p)
{
    return &block.data[0] == p;
}
#endif



template <typename Block, typename OtherBlock>
void test()
{
    double *p;
    Block A(3);
    A.data[0]= 5.0;    
   
    cout << "A= f(A, p);\n";
    A= f(A, p);

    if (A.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (compare(A, p)) 
	throw "Block is not moved/copied appropriately!";

    cout << "Block B= f(A, p);\n";
    Block B= f(A, p);

    if (B.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    // There seems to be never a copy
    if (&B.data[0] != p) 
	throw "This is the first time that an expression in a constructor is copied!";


    // This type is guarateed to be different to f's return type
    // In this case the block MUST be copied
    OtherBlock C(3);

    cout << "C= f(A, p);\n";
    C= f(A, p);

    if (C.data[0] != 5.0) 
	throw "Wrong value trying to move, should be 5.0!";
    if (&C.data[0] == p) 
	throw "Block must be copied not moved!";

}




int test_main(int argc, char* argv[])
{

    typedef contiguous_memory_block<double, false, 0>  dblock;
    typedef contiguous_memory_block<double, true, 3>   sblock;

    test<dblock, sblock>();
    test<sblock, dblock>();

#if 0

    typedef generic_array<double, false, 0>            darray;
    typedef generic_array<double, true, 3>             sarray;

    test<darray, sarray>();
    test<sarray, darray>();

#endif	

    return 0;
}
