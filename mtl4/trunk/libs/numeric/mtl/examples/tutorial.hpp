// $COPYRIGHT$

#ifndef MTL_TUTORIAL_INCLUDE
#define MTL_TUTORIAL_INCLUDE


// This file contains no source code but only documentation.

/*! \mainpage MTL4 manual

\author Peter Gottschling
\date June 2007

The Matrix Template Library (incarnation) 4 is a generic library for linear
algebra operations on matrices and vectors.
Its goal is to facilitate its usage comparable to mathematical libraries
like Mathematica and Matlab and to approach, at the same time, performance 
characteristics of high-performance libraries like BLAS or ATLAS.
In fact, programs can be written in a natural operator notation and the 
library can evaluate the expressions with an optimized library.
However, this is limited to types that are supported by these libraries.
An important distinction to BLAS is that sparse matrices are supported.

- \subpage intro 
- \subpage install 
- \subpage tutorial  
*/

//-----------------------------------------------------------

/*! \page intro Introduction


The Matrix Template Library (incarnation) 4 is a generic library for linear
algebra operations on matrices and vectors.
Its goal is to facilitate its usage comparable to mathematical libraries
like Mathematica and Matlab and to approach, at the same time, performance 
characteristics of high-performance libraries like BLAS or ATLAS.
In fact, programs can be written in a natural operator notation and the 
library can evaluate the expressions with an optimized library.
However, this is limited to types that are supported by these libraries.
An important distinction to BLAS is that sparse matrices are supported.




Proceed to the \ref install "installation guide".

*/

//-----------------------------------------------------------


/*! \page install Installation guide

MTL4 is a pure template library and only a download of the sources
is required.

The <a href="http://www.boost.org">Boost library</a>
is used and must also be downloaded.
We used in the development and testing version 33.1 but the programs
would probably compile with earlier versions, too.
 The parts of boost used in MTL4 do not need
to be compiled but only included.

If you want to run the test programs, you need the build system
<a href="http://www.scons.org">scons</a>.
It is easy to install and takes only a few minutes.
The scons-based build of MTL4 uses the environment variables 
<tt>MTL_BOOST_ROOT</tt> to locate the MTL directory
and <tt>BOOST_ROOT</tt> to locate the Boost directory.
The path of the MTL directory is normally not needed if you do not
have version control with multiple development branches for MTL4.


To execute the test programs go in MTL4's test directory
libs/numeric/mtl/test and type:\n
<tt>scons -D . check=1</tt>\n
If the building finishes all tests were passed.
The building can be considerably speed up, esp. on multi-core processors,
when scons is used with multiple processes.
For instance, to run the tests with four processes (which works quite
well on two processors) type:\n
<tt>scons -Dj 4 . check=1</tt>\n
The output will be quite chaotic but, again, when the building finishes
all tests are passed.

Similarly, the example programs can be compiled.
Go in directory libs/numeric/mtl/examples and type:\n
<tt>scons -D .</tt>\n
For the sake of simplicity, there are no checks in the examples.

If you want to use BLAS, you need to define the macro <tt>MTL_HAS_BLAS</tt>,
e.g., by compiling your programs with 
<tt>-DMTL_HAS_BLAS</tt>, and link the appropriate libraries.
Alternatively, you can use MTL4's build system 
with the flag <tt>with-blas=1</tt> that will
check if GotoBlas, ACML, or ATLAS is installed on your system
(thanks to Torsten Hoefler who wrote the tests in scons).
If scons does not find your BLAS library you can specify additional
flags, see\n
<tt>scons -h</tt>\n
for details.

If you wish to generate the documentation locally on your system
you need <a href="http://www.doxygen.org">doxygen</a>.
When it is installed type <tt>doxygen</tt> in the main directory and
the documentation will be written to libs/numeric/mtl/doc.

Resuming, for MTL4 you need to:
- Include the MTL path;
- Include the boost path;
- Optionally install scons;
- Optionally install a BLAS library; and
- Optionally install doxygen.


Proceed to the \ref tutorial "the tutorial".  

*/


//-----------------------------------------------------------


//-----------------------------------------------------------

/*! \page tutorial Tutorial

This tutorial introduces the user into:

-# \subpage vector_def
-# \subpage vector_functions
-# \subpage vector_expr 



*/





//-----------------------------------------------------------


/*! \page vector_def Vector definitions

To start the tutorial we want to give a very short example (we could call
it the MTL4-hello-world).

\include vector1.cpp

The <a href="http://www.boost.org">Boost library</a>
is used and must also be downloaded. See the
\ref install "installation guide" for more details.
To compile a MTL4 program you only need to include the MTL and the
boost path.
A compile command could read:\n 
<tt>g++ -I/u/peter/mtl4 -I/u/peter/boost -O2 vector1.cpp -o vector1</tt>\n
As most modern C++ software MTL4 uses intensively function inlining.
As a consequence, the performance is rather poor if compiled without
optimization.
But don't worry: despite the aggressive source code transformation at 
compile time, the compilation rarely took more than a minute, in
most cases only a few seconds.

The short program certainly does not need much explanation only some
brief comments.
The vector in the program above is a column vector.
The constructor in the example takes two arguments: the size and the 
initial value.

Indices always start with zero.
Earlier efforts to support one-based indices were abondoned because
code became rather complicated when mixed indexing for different
arguments of a function.
We decided that the additional development
 effort and the potential performance
penalty are not acceptable.
Extra functionality will be provided in the future if necessary for 
interoperability with Fortran libraries.

The following program defines a row vector of 7 elements without 
(explicit) initialization.

\include vector2.cpp

Scalar values can be assigned to vectors if the type of the scalar
value is assignable to the type of the elements.
Scalar types are in MTL4 all types that are not explicitly defined
by type traits as vectors or matrices, thus almost all types.


Proceed to \ref vector_functions "vector functions".  

*/

//-----------------------------------------------------------


/*! \page vector_functions Vector functions


Principal MTL4 functions are all defined in namespace mtl.
Helper functions are defined in sub-namespaces to avoid
namespace pollution.

The following program shows how to compute norms:

\include vector_norm.cpp

Since this code is almost self-explanatory, we give only a few
comments here.
Vector norms are for performance reasons computed with unrolled loops.
Since we do not want to rely on the compilers' capability and 
in order to have more control over the optimization, the unrolling
is realized by meta-programming.
Specializations for certain compilers might be added later
if there is a considerable performance gain over the meta-programming
solution.

Loops in reduction operations, like norms, are by default unrolled
to 8 statements.
The optimal unrolling depends on several factors, in particular
the number of registers and the value type of the vector.
The last statement shows how to unroll the computation to six
statements.
The maximum for unrolling is 8 (it might be increased later).

The norms return the magnitude type of the vectors' value type, 
see Magnitude.

Similarly, the sum and the product of all vector's elements can
be computed:

\include vector_reduction.cpp

Proceed to \ref vector_expr "vector expressions".  

*/

//-----------------------------------------------------------


/*! \page vector_expr Vector expressions



*/



#endif // MTL_TUTORIAL_INCLUDE
