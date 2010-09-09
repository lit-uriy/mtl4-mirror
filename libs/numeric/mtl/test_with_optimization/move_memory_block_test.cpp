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

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


// Everything in the test is double


using namespace std;
using namespace mtl;
using detail::contiguous_memory_block; 

typedef contiguous_memory_block<double, false, 0>  dblock;
typedef contiguous_memory_block<double, true, 3>   sblock;



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
bool compare(const dblock& block, double* p)
{
    return &block.data[0] != p;
}

// For blocks on stack, equal addresses means accidental moving 
bool compare(const sblock& block, double* p)
{
    return &block.data[0] == p;
}


template <typename Block>
void print(const Block& block, double* p)
{
    cout << "Data was " << (&block.data[0] == p ? "moved.\n" : "copied.\n");
}

template <typename Block, typename OtherBlock>
void test()
{
    double *p;
    Block A(3);
    A.data[0]= 5.0;    
   
    cout << "A= f(A, p);\n";
    A= f(A, p);
    print(A, p);

    if (A.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (compare(A, p)) 
	throw "Block is not moved/copied appropriately!";

    cout << "Block B= f(A, p);\n";
    Block B= f(A, p);
    print(B, p);

    if (B.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    // There seemed to be never a copy, now static arrays are copied
    if (&B.data[0] != p) 
	throw "This is the first time that an expression in a constructor is copied!";
#if 0
    if (compare(B, p)) 
	throw "Block is not moved/copied appropriately!";
#endif


    // This type is guarateed to be different to f's return type
    // In this case the block MUST be copied
    OtherBlock C(3);

    cout << "C= f(A, p);\n";
    C= f(A, p);
    print(C, p);

    if (C.data[0] != 5.0) 
	throw "Wrong value trying to move, should be 5.0!";
    if (&C.data[0] == p) 
	throw "Block must be copied not moved!";

    // New block, in this case the block MUST be copied
    OtherBlock D(A);

    cout << "D(A);\n";
    print(C, &A.data[0]);

    if (D.data[0] != 5.0) 
	throw "Wrong value in copy constructor, should be 5.0!";
    if (&D.data[0] == &A.data[0]) 
	throw "Block must be copied not moved!";

}

enum e_t {own_e, external_e, view_e};

void dynamic_test(dblock& block, e_t e, const char* name)
{
    cout << '\n' << name;

    dblock A= block;
    cout << "dblock A(block)\n";
    print(A, &block.data[0]);

    if ((e == view_e) ^ (&block.data[0] == &A.data[0]))
	throw "Only views have shallow semantics.\n";


    double *p;
    
    cout << "block= f(A, p);\n";
    block= f(A, p);
    print(block, p);

    if ((e == own_e) ^ (&block.data[0] == p))
	throw "Only blocks with their own data can move results.\n";

    cout << "dblock B(clone(block));\n";
    dblock B= clone(block);
    print(B, &block.data[0]);

    if (&block.data[0] == &B.data[0])
	throw "Cloned blocks must all be copied.\n";

}


int test_main(int argc, char* argv[])
{

    cout << "Copy/heap data in heap / other block on stack.\n";
    test<dblock, sblock>();
    cout << "\nCopy data in stack / other block on heap.\n";
    test<sblock, dblock>();

    dblock    own(3), fixed(3), external(&fixed.data[0], 3), view(&fixed.data[0], 3, true);
    // Store a value hoping for avoiding to optimize allocations away
    own.data[0]= 7.0; external.data[0]= 7.0; view.data[0]= 7.0;
    cout << "Adresses: " << own.data << ", " << external.data << ", " << view.data << "\n";

    dynamic_test(own, own_e, "Own data\n");
    dynamic_test(external, external_e, "External data\n"); 
    dynamic_test(view, view_e, "View on own\n");

    return 0;
}
