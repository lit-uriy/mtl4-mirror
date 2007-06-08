// $COPYRIGHT$

#ifndef MTL_TUTORIAL_INCLUDE
#define MTL_TUTORIAL_INCLUDE

// for references
namespace mtl {

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
-# \subpage rich_vector_expr 
-# \subpage matrix_types
-# \subpage matrix_insertion



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
The definitions of the \ref one_norm, \ref two_norm, and 
\ref infinity_norm can
be found in their respective documentations.
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

As vector reductions base on the same implementation as norms, the
unrolling can be explicitly controlled as shown in the last
command.
The results of these reductions are the value type of the vector.

The dot product of two vectors is computed with the function \ref dot:

\include dot.cpp

As the previous computation the evaluation is unrolled, either with
a user-defined parameter or by default eight times.

The result type of \ref dot is of type of the values' product.
If MTL4 is compiled with a concept-compiler, the result type is 
taken from the concept std::Multiple and without concepts
Joel de Guzman's result type deduction from Boost is used.

Proceed to \ref vector_expr "vector expressions".  

*/

//-----------------------------------------------------------


/*! \page vector_expr Vector expressions

The following program illustrates the usage of basic vector
expressions.

\include vector_expr.cpp

The mathematical definition of vector spaces requires that
vectors can be added, multiplied with scalar values
and the results can be assigned to vectors.
In MTL4, the vectors must have the same algebraic shape, 
see \ref ashape,
 for addition
and assignment, i.e. column vectors cannot be assigned to row vectors.
If the elements of the vectors are vectors themselves or matrices
then the elements must also be of the same algebraic shape.

Products of scalars and vectors are
 implemented by a view, see \ref vector::scaled_view,
and vector elements are multiplied with the factor when
accessing an element of the view.
Please notice that the scaling factor's type is not required to be
identical with the vector's value type.
Furthermore, the value type of the view can be different from
the vector's value type if necessary to represent the products.
The command is an example for it: multiplying a double vector
with a complex number requires a complex vector view to 
guarantee the correctness of the results.

Traditional definitions of operators perform computations
in temporary variables that are returned at the end of the
calculation.
The presence of multiple operators, say n, in a single expression
(which is always the case except for an assignment without numerics)
requires then the execution of n loops (possibly more to copy
the temporaries on the stack).
If the vectors are too large for the cache, values must be loaded
repeatingly from slower memories.
Expression templates circumvent this repeated loading of vector
elements by
performing only one loop.


Proceed to \ref rich_vector_expr "rich vector expressions".  

*/

//-----------------------------------------------------------


/*! \page rich_vector_expr Rich vector expressions

As discussed in the previous chapter, 
vector operation can be accelerated by improving
their cache locality via expression templates.
Cache locality can be further improved in applications
when subsequent vector expressions are evaluated
in one loop, data dependencies allowing.
Unfortunately, this so-called loop fusion cannot be 
realized with expression templates.
At least not when the loops are performed in the assignment.

In colloboration with Karl Meerbergen, we developed expression
templates that can be nested, called rich expression templates.
The following program shows some examples of rich expression
templates:

\include rich_vector_expr.cpp

The first example shows the combination of an incremental
assignment with a vector addition.
The second statement fuses four vector expressions:
-# The value 2 is assigned to every element of x;
-# w is scaled in-place with 3;
-# v is incremented by the sum of both vector; and
-# u is incremented by the new value of v.

Again, all these operations are performed in one loop and each vector
element is acceessed exactly once.


Proceed to \ref matrix_types "matrix types".  

*/

//-----------------------------------------------------------


/*! \page matrix_types Matrix types

Right now, MTL4 provides three matrix types:
- \ref dense2D;
- \ref morton_dense; and
- \ref compressed.

The type \ref dense2D defines regular 
row-major and column-major matrices:

\include dense2D.cpp

If no matrix parameters are defined, dense matrices are
by default row-major.
There are more matrix parameters besides the orientation.
As they are not yet fully supported we refrain from discussing
them.

Matrix elements can be accessed by a(i, j) or in the more
familiar form a[i][j].
The second form is internally transformed into the first
one at compile time so that the run-time performance is not
affected (unless the compiler does not inline completely
which we never observed so far).
Also, the compile time is not conceivably increased by this
transformation.

Please notice that overwriting single matrix elements is only
defined for dense matrix types. 
For a generic way to modify matrices see \ref matrix_insertion.

Assigning a scalar value to a matrix stores a multiple of
the identity matrix, i.e. the scalar is assigned to all
diagonal elements and all off-diagonal elements are 0.
If the matrix is not square this assignment throws an exception.
By the way, this operation is generic (i.e. applicable to
all matrix types).

Dense matrices with a recursively designed memory layout
can be defined with the type \ref morton_dense:

\include morton_dense.cpp

A detailed description will be added soon.

Sparse matrices are defined with the type \ref compressed2D:

\include compressed2D.cpp

Matrix a is stored as compressed row storage (CRS).
Its assigned values correspond to a discretized Laplace operator.
To change or insert single elements of a compressed matrix
is not supported
Especially for very large matrices, this would result in an
unbearable performance burden.

However, it is allowede to assign a scalar value to the entire matrix
given it is square as in the example.
Matrix b is stored in compressed column storage (CCS).

Which orientation is favorable dependents on the performed
operations and might require some experimentation.
All operations are provided in the same way for both formats

How to fill  sparse matrices is shown in the following chapter.


Proceed to \ref matrix_insertion "matrix insertion".  

*/

//-----------------------------------------------------------


/*! \page matrix_insertion Matrix insertion


*/

} // namespace mtl

#endif // MTL_TUTORIAL_INCLUDE
