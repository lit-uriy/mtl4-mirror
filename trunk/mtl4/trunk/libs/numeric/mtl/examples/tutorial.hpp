// $COPYRIGHT$

#ifndef MTL_TUTORIAL_INCLUDE
#define MTL_TUTORIAL_INCLUDE

// for references
namespace mtl {

// This file contains no source code but only documentation.

/*! \mainpage MTL4 manual

\author Peter Gottschling and Andrew Lumsdaine

The %Matrix Template Library (incarnation) 4 is a generic library for linear
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


The %Matrix Template Library (incarnation) 4 is a generic library for linear
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

MTL4 is still in an early state of development and it is possible that
some details may change during further implementation.
However, we will do our best that applications are minimally affected.
In particular, the topics in the tutorial are not subject to modifications.
This, of course, does not exclude backward-compatible extensions.

-# Basic Types and Operations
   -# \subpage vector_def
   -# \subpage vector_functions
   -# \subpage vector_expr 
   -# \subpage rich_vector_expr 
   -# \subpage matrix_types
   -# \subpage matrix_insertion
   -# \subpage matrix_functions
   -# \subpage matrix_expr 
   -# \subpage matrix_vector_functions 
   -# \subpage matrix_vector_expr
   .
-# Traversal of Matrices and Vectors
   -# \subpage iteration
   -# \subpage rec_intro
   .
-# Advanced Topics
   -# \subpage function_nesting

*/





//-----------------------------------------------------------


/*! \page vector_def Vector Definitions

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
Earlier efforts to support one-based indices were abandoned because
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


/*! \page vector_functions Vector Functions


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

The example also showed how to compute the conjugate values of all
elements.
The vector is not changed but a view on the vector is created
that conjugate an element when accessed.

The transposition of vectors will be implemented soon.

Return to \ref vector_def "vector definitions"
or proceed to \ref vector_expr "vector expressions".


*/

//-----------------------------------------------------------


/*! \page vector_expr Vector Expressions

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
repeatedly from slower memories.
Expression templates circumvent this repeated loading of vector
elements by
performing only one loop.


Return to \ref vector_functions "vector functions"
or proceed to \ref rich_vector_expr "rich vector expressions".

*/

//-----------------------------------------------------------


/*! \page rich_vector_expr Rich Vector Expressions

As discussed in the previous chapter, 
vector operation can be accelerated by improving
their cache locality via expression templates.
Cache locality can be further improved in applications
when subsequent vector expressions are evaluated
in one loop, data dependencies allowing.
Unfortunately, this so-called loop fusion cannot be 
realized with expression templates.
At least not when the loops are performed in the assignment.

In collaboration with Karl Meerbergen, we developed expression
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
element is accessed exactly once.


Return to \ref vector_expr "vector expressions"
or proceed to \ref matrix_types "matrix types".

*/

//-----------------------------------------------------------


/*! \page matrix_types Matrix Types

Right now, MTL4 provides three %matrix types:
- \ref dense2D;
- \ref morton_dense; and
- \ref compressed.

The type \ref dense2D defines regular 
row-major and column-major matrices:

\include dense2D.cpp

If no %matrix parameters are defined, dense matrices are
by default row-major.
There are more %matrix parameters besides the orientation.
As they are not yet fully supported we refrain from discussing
them.

%Matrix elements can be accessed by a(i, j) or in the more
familiar form a[i][j].
The second form is internally transformed into the first
one at compile time so that the run-time performance is not
affected (unless the compiler does not inline completely
which we never observed so far).
Also, the compile time is not conceivably increased by this
transformation.

Please notice that overwriting single %matrix elements is only
defined for dense %matrix types. 
For a generic way to modify matrices see \ref matrix_insertion.

Assigning a scalar value to a %matrix stores a multiple of
the identity %matrix, i.e. the scalar is assigned to all
diagonal elements and all off-diagonal elements are 0.
If the %matrix is not square this assignment throws an exception.
By the way, this operation is generic (i.e. applicable to
all %matrix types).

Dense matrices with a recursively designed memory layout
can be defined with the type \ref morton_dense:

\include morton_dense.cpp

A detailed description will be added soon.

Sparse matrices are defined with the type \ref compressed2D:

\include compressed2D.cpp

%Matrix a is stored as compressed row storage (CRS).
Its assigned values correspond to a discretized Laplace operator.
To change or insert single elements of a compressed %matrix
is not supported.
Especially for very large matrices, this would result in an
unbearable performance burden.

However, it is allowed to %assign a scalar value to the entire %matrix
given it is square as in the example.
%Matrix b is stored in compressed column storage (CCS).

Which orientation is favorable dependents on the performed
%operations and might require some experimentation.
All %operations are provided in the same way for both formats

How to fill  sparse matrices is shown in the following chapter.


Return to \ref rich_vector_expr "rich vector expressions"
or proceed to \ref matrix_insertion "matrix insertion".

*/

//-----------------------------------------------------------


/*! \page matrix_insertion Matrix Insertion

Setting the values of a dense %matrix is an easy task since each element
has its dedicated location in memory.
Setting sparse matrices, esp. compressed ones, is a little more complicated.
There exist two extreme approaches:
- Inserting all values on the fly at any time; or
- Providing a special insertion phase and then creating the compressed format
  once and forever.

The former approach has the advantage that it is handier and that the set-up
of sparse matrices can be handled like dense matrices (which eases the development
of generic code).
However, when matrices grow larger, the insertion becomes more and more expensive,
up to the point  of being unusable.
Most high-performance libraries use therefore the second approach.
In practice, a sparse %matrix is usually the result of discretization (FEM, FDM, ...)
that is set up once and then used many times in linear or non-linear solvers.
Many libraries even establish a two-phase set-up: first building the sparsity pattern
and then populating the non-zero elements with values.

The MTL4 approach lies somewhere between.
Sparse matrices can be either written (inserted) or read.
However, there can be multiple insertion phases.

Before giving more details, we want to show you a short example:

\include insert.cpp

The first aspect worth pointing at is that sparse and dense matrices are treated
the same way.
If we cannot handle sparse matrices like dense (at least not efficiently), we
can treat dense matrices like sparse ones.
For performance reasons, matrices are not initialized by default. 
Therefore, the first operation in the function fill is to set the matrix to zero.


Internally the inserters for dense and sparse matrices are implemented completely
differently but the interface is the same.
Dense inserters insert the value directly and there is not much to say about.

Sparse inserters are more complicated.
The constructor stretches the matrix so that the first five elements in a row
(in a CCS matrix likewise the first 5 elements in a column) are inserted directly.
During the live time of the inserter, new elements are written directly into
empty slots. 
If all slots of a row (or column) are filled, new elements are written into an std::map.
During the entire insertion process, no data is shifted.

If an element is inserted twice then the existing element is overwritten, regardless
if the element is stored in the matrix itself or in the overflow container.
Overwriting is only the default. The function modify() illustrates how to use the inserter
incrementally.
Existing elements are incremented by the new value.
We hope that this ability facilitates the development of FEM code.

For performance reasons it is advisable to customize the number of elements per row (or column),
e.g., ins(m, 13).
Reason being, the overflow container consumes  more memory per element then the 
regular matrix container.
In most applications, an upper limit can be easily given.
However, the limit is not that strict: if some rows need more memory than the slot size it only
results in slightly higher memory need for the overflow container.
If the number of elements per row is very irregular we recommend a slot size over the average
(and maybe under the maximum).
Since only a small part of the data is  copied during the compression, sparse matrices 
can be created that fill almost the entire memory.

Nota bene: inserters for dense matrices are not much more than facades for the matrices themselves
in order to provide the same interface as for sparse ones.
However, dense inserters can be also very useful in the future for extending the 
library to parallel computations.
Then the inserter can be used to write values into remote matrix elements.

A more powerful method to fill sparse (and dense) matrices provide the two functions
element_matrix() and element_array().

The following program illustrates how to use them:

\include element_matrix.cpp

The function element_array is designed for element matrices that are stored as 
a 2D C array.
The entries of such an element %matrix are accessed by A[i][j],
while the entries are accessed by A(i, j) if the function element_matrix is used.
Element matrices stored in MTL4 types can be accessed both ways and either
element_array or element_matrix can be used.

Both functions can be called with two or three arguments.
In the former case the first argument is the element matrix and the second argument
a vector containing the indices that correspond to the rows and columns of the
assembled matrix.
With three arguments, the second one is a vector of row indices and the third one
a vector with column indices.
Evidently, the size of the vector with the row/column indices should be equal to the
number of rows/columns of the element matrix.

The vector type must provide a member function size and a bracket operator.
Thus, mtl::dense_vector and std::vector can used (are models).



Return to \ref matrix_types "matrix types"
or proceed to \ref matrix_functions "matrix functions".

*/

//-----------------------------------------------------------


/*! \page matrix_functions Matrix Functions

Norms on matrices can be computed in the same fashion as on vectors:

\include matrix_norms.cpp

Other matrix functions compute trace of a matrix,
the element-wise conjugates, and the transposed matrix:

\include matrix_functions2.cpp

The functions conj(A) and trans(A) do not change the matrices
but they return views on them.

More functions will be implemented in the future.


Return to \ref matrix_insertion "matrix insertion"
or proceed to \ref matrix_expr "matrix expressions".


*/

//-----------------------------------------------------------


/*! \page matrix_expr Matrix Expressions


The following program illustrates how to add matrices, including scaled matrices:

\include matrix_addition.cpp

The example shows that arbitrary combinations of matrices can be added, regardless their
orientation, recursive or non-recursive memory layout, and sparseness.

Matrix multiplication can be implemented as elegantly:


\include matrix_mult_simple.cpp

Arbitrary matrix types can be multiplied in MTL4.
Although inefficient combinations might be implemented inefficiently--we 
come back to this later.
Let's start with the operation that is the holy grail in 
high-performance computing:
dense matrix multiplication.
This is also the operation shown in the example above.
The multiplication  can be executed with the function mult
where the first two arguments are the operands and the third the result.
Exactly the same is performed with the operator notation below.

Warning: the arguments and the result must be different!
Expressions like A= A*B will throw an exception.
More subtle aliasing, e.g., partial overlap of the matrices
might not be detected and result in undefined mathematical behavior.

Products of three matrices are not supported.
However, two-term products can be arbitrarily added and subtracted:

\include matrix_mult_add.cpp

Return to \ref matrix_functions "matrix functions"
or proceed to \ref matrix_vector_functions "matrix-vector functions".

*/

//-----------------------------------------------------------


/*! \page matrix_vector_functions Matrix-Vector Functions

Available matrix-vector operations are currently multiplication
and updates.
The former can also be expressed by operator and is described
as \ref matrix_vector_expr expression.
The application of rank-one and rank-two updates are
illustrated in the following (hopefully self-explanatory)
program:

\include rank_two_update.cpp

The output of the matrix is formatted for better readability.
The functions also work for sparse matrices although we
cannot recommend this for the sake of efficiency.

In the future, updates will be also expressible with operators.
For instance, rank_one_update(A, v, w) can be written as
A+= conj(v) * trans(w) if v and w are column vectors (if w
is a row vector the transposition can-and must-be removed).
Thus, the orientation is relevant in operator notation
where the functions rank_one_update and rank_two_update
ignore the orientation.

Return to \ref matrix_expr "matrix expressions"
or proceed to \ref matrix_vector_expr "matrix-vector expressions".


*/

//-----------------------------------------------------------


/*! \page matrix_vector_expr Matrix-Vector Expressions


Matrix-vector products are written in the natural way:

\include matrix_vector_mult.cpp

The example shows that sparse and dense matrices can be multiplied
with vectors.
For the sake of performance, the products are implemented with 
different algorithms.
The multiplication of Morton-ordered matrices with vectors is
supported but currently not efficient.

As all products the result of a matrix-vector multiplication can be 
 -# Directly assigned;
 -# Incrementally assigned; or
 -# Decrementally assigned (not shown in the example).
.
to a vector variable.

Warning: the vector argument and the result must be different!
Expressions like v= A*v will throw an exception.
More subtle aliasing, e.g., partial overlap of the vectors
might not be detected and result in undefined mathematical behavior.

Matrix-vector products (MVP) cannot be combined with arbitrary vector
operations.
It is planned for the future to support expressions like
r= b - A*x.

Already supported is scaling of arguments, as well for the matrix
as for the vector:

\include scaled_matrix_vector_mult.cpp

All three expressions and the following two blocks
compute the same result.
However, since vector elements are accessed multiple times in an MVP
it is inefficient to scale the vector
and obviously it is an even bigger waste to  scale both
arguments.

The matrix scaling on the fly requires the same number of operations
as an in-place scaling with subsequent MVP (as each matrix element is
used only once).
Thus, it is advisable to scale the matrix in the expression instead
of scaling it before the multiplication and scaling it back afterwards
(as the fourth block of operations).
Scaling the vector upfront is more efficient under the quite likely
assumption that A has more than 2*n non-zero elements.

Return to \ref matrix_vector_functions "matrix-vector functions"
or proceed to \ref iteration "iteration".


*/

//-----------------------------------------------------------


/*! \page iteration Iteration

This section will be written soon.

Return to \ref matrix_vector_expr "matrix-vector expressions"
or proceed to \ref rec_intro "recursion".


*/

//-----------------------------------------------------------


/*! \page rec_intro Recursion


Recursion is an important theme in MTL4.
Besides matrices with recursive recursive memory layout -- cf. \ref matrix_types and \ref morton_dense --
recursion with regard to algorithms plays a decisive role.

To support the implementation of recursive algorithms we introduced -- in collaboration with David S. Wise --
the concept to Recursator, an analogon of <a href=" http://www.sgi.com/tech/stl/Iterators.html">Iterator</a>.
The class matrix_recursator enables recursive subdivision of all matrices with a sub_matrix function
(e.g., dense2D and morton_dense).
We refrained from providing the sub_matrix functionality to compressed2D; this would possible but very inefficient
and therefor not particularly useful.
Thus matrix_recursator of compressed2D cannot be declared.
A recursator for vectors is planned for the future.

Generally spoken, the matrix_recursator (cf. \ref recursion::matrix_recursator)
consistently divides a matrix into four quadrants 
- north_west;
- north_east;
- south_west; and
- south_east;
.
with the self-evident cartographic meaning (from here on we abreviate matrix recursator to recursator).
The quadrants itself can be sub-divided again providing the recursive sub-division of matrices
into scalars (or blocks with user-defined maximal size).

The following program illustrates how to divide matrices via recursator:

\include recursator.cpp

The functions north_west(), north_east(), south_west(), and south_east()  return recursators
that refer to sub-matrices.
The sub-matrices can be accessed by dereferring the recursator, i.e. *rec.
Only then a sub-matrix is created. 

As the example shows, the quadrant (represented by a recursator) can be sub-divided
further (returning another recursator).
Block-recursive algorithms can be implemented efficiently by sub-dividing large matrices
recursively into blocks of decreasing size until a block size is reached that allows efficient
iterative treatment.
Sub-matrices are only created at the base case and not during the recursive descent
because the creation of sub-matrix might be a relatively expensive operation (e.g., with morton_dense) 
while the creation of a new recursator requires only a few integer operations.

The recursator uses internally a virtual bound that is a power of 2 and at least as large as
the number of rows and columns.
In the example, the bound is 16 (as shown by the member function bound).
When computing a quadrant the bound is halved and the starting row and column are potentially increased.
For instance, the north_east quadrant is a virtual 8 by 8 matrix starting at row 0 and column 8.
The sub-matrix referred by the north_east recursator is the intersection of this virtual quadrant with
the original matrix A, i.e. an 8 by 2 matrix starting in row 0 and column 8.

More functionality of recursators is shown in the following example:

\include recursator2.cpp

The function is_empty applied on a recursator computes whether the referred sub-matrix is empty,
i.e. the intersection of the virtual quadrant and the original matrix A is empty.
The sub-matrix itself is not generated since this test can be performed from size and index information.
In the same way, number of rows and columns of the referred sub-matrix can be computed without its creation.

The function is_full() comes in handy in block-recursive algorithms.
Assume we have a base case of 64 by 64, i.e. matrices with at most 64 rows and columns are treated iteratively.
Then it is worthwile to write a blazingly fast iterative implementation  for 64 by 64 matrices,
in other words when the sub-matrix fills the entire virtual quadrant (when bound is 64).
Thus, the function is_full() can be used to dispatch between this optimized code and the (hopefully not
much slower) code for smaller matrices.


Return to \ref iteration "iteration"
or proceed to \ref function_nesting "why and how we use functors".


*/

//-----------------------------------------------------------


/*! \page function_nesting Why and How we use Functors

The standard user interface of MTL4 consists of functions and operators.
Internally these functions are often implemented by means of functors.
This has two reasons. The first reason is that functions cannot be partially specialized
(cf. <a href="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2001/n1295.asc">Document 
number J16/01-0009 = WG21 N1295 from the C++ Standard Committee</a>)
and the second reason is that functors allow an arbitrary composition.
We illustrate in this section how function-like interfaces can be enriched by partial
specialization and composition.

Assume we want to write a templated multiplication function for matrices:

\include nesting/function.hpp

Dense matrix multiplication is the first operation where all the techniques on
this page are applied.
Of course it is planned to extend other operations in the same manner.



\section functor_sec1 Step 1: Transform a Function into a Functor

We replace this function by a class containing an application operator
with the same signature:

\include nesting/functor.hpp

An object of this class

\include nesting/functor_obj.hpp

can be called like a function. Admittedly, the definition of this functor does not look very elegant.
Nevertheless, it is necessary to provide composition and partial specialization whereby the impact for
the user can be minimized by the techniques described below.

Remark: the suffix "_ft" stands for fully templated, in contrast to functor classes where all or part of
the types are automatically instantiated, as shown in step x.


\section functor_sec2 Step 2: Template Specialization

After the functor is implemented with a default behavior, one can write specializations for a certain
type or like in our case a certain combination of types:

\include nesting/special_functor.hpp

Please note that specializations are not required to be written in the same file as the template function
(i.e. by the same author) but can be added in any file that is included in the compilation unit.

By the way, this explicit form of specialization is also supported for functions (but the following 
techniques are not).


\section functor_sec3 Step 3: Partial Specialization

Very often specializations are not only possible for one single type (or tuple of types) but for an entire
set of types.
If, for instance, a more efficient implementation of mult is available for arbitrary triplets of dense2D matrices
regardless their respective value types and parameters, the functor can be partially specialized:

\include nesting/partial_functor.hpp

Again, such specializations can be added later. 
This becomes very handy when users define their own (matrix) types and 
can also provide specialized implementations for certain functions or operators
which are implemented in terms of functors.


\section functor_sec4 Step 4: Reuse of Functors 


Assume we want implement a functor that multiplies matrices using BLAS routines.
We know upfront that only a few type triplets are supported and all other matrix types
need another implementation.
One solution to implement such a functor is to call by default an already implemented
function and specialize this functor for certain type typles:

\include nesting/blas_functor_ugly.hpp

This code works but we can write it more elegantly with public inheritence:

\include nesting/blas_functor.hpp

This program is not only shorter but can eventually reduce the compilation cost,
for details look in David Abraham's book for meta-function forwarding. 


\section functor_sec5 Step 5: Conditional Specialization


This is only a small change but it can make a conceivable difference.
BLAS routines impressingly fast but we do not want to require mandatorily BLAS to be installed.
Guarding the specializations with configuration-dependent macros allows us to provide
the BLAS functions only when they are available.

\include nesting/blas_functor_cond.hpp

In case BLAS is not installed in MTL4, the programs calling the BLAS functor 
still work (not necessarily as fast).

In fact if you call an MTL4 functor, you are guaranteed that the operation is 
correctly performed.
If a functor with an optimized implementation cannot handle a certain type tuple,
it calls another functor that can handle it (otherwise calls yet another functor in turn
that can perform the operation (otherwise ...)).




\section functor_sec6 Step 6: Functor Composition


Resuming the previous sections, we can define a default behavior and one or more
specialized behaviors for a template functor.
Now we like to costumize the default behavior of functors.

The only thing we need to do for it is to introduce a template parameter for
the default functionality:


\include nesting/blas_functor_comp.hpp

The parameter for the default functor can of course have a default value, as in the example.
The name "Backup" is understood that the functors implement a functionality for a certain
set of type tuples.
Type tuples that are not in this set are handled by the Backup functor.
Theoretically, such functors can be composed arbitrarily.
Since this is syntantically somewhat cumbersome we will give examples later.


\section functor_sec7 Step 7: Functors with Automatic Instantiation

The usage of functors had two purposes: the partial specialization and the composition.
The former requires all types to be template arguments while the composition
does not.
Therefore we introduce another category of functors where the function arguments 
are not template arguments.
These functors (more precisely their operators) call the fully templated functors
to not loose the capability of partial specialization:

\include nesting/blas_functor_auto.hpp

Before we finally come to some examples we want to introduce another template
parameter.
This leads us to the actual implemenation of the functors, 
for instance the BLAS functor:

\include nesting/blas_functor_mtl.hpp

The parameter Assign allows the realization of C= A*B, C+= A*B, and C-= A*B with the
same implementation (an explanation will follow).
At this point we focus on the composition.

The duality of fully and partially templated functors simplifies the syntax of composed
functors significantly.
Already the default type of the backup functor can benefit from the shorter syntax
as shown in the example above.


\section functor_avail Available Functors


MTL4 provides several functors for dense matrix multiplication:
-# Canonical implementation with 3 nested loops and iterators;
-# A corresponding 3-loop implemtation with cursors and property maps;
-# Tiled products for regular matrices using pointers with
   -# With tile size 2 by 2;
   -# With tile size 4 by 4; and 
   -# Costumizable tile size;
   .
-# Recursive matrix product with costumizable base case (kernel);
-# Platform optimized implementation; and
   -# So far only one implementation from Michael Adams for Opteron
   .
-# BLAS functor calling the corresponding routines.

All these functors have a Backup parameter which is by default set to 
the canonical implementation with iterators.
The two canonical products support all combination of matrix types
and their Backup parameter is only added to unify the interface.

\section functor_example Functor Composition Example

As an example, we want to define a functor that calls:
- BLAS if available, otherwise
- The platform-specific code if available, otherwise
- The 4 by 4 tiled product, otherwise
- The canonical implementation.

The Backup parameter needs only be set if another then the canonical implementation
is used.
If you use typedefs it is advisable to work from buttom up through the list:
The tiled 4 by 4 product has already the right defaults.
The platform-specific version needs a non-default backup parameter.
This requires also the definition of the assign parameter because it is
positioned before.
We keep this combined functor type as a type definition and use
it finally in the BLAS functor.
Here we create directly an object of this type which can be later called like a function:

\include nesting/comp_example.hpp

Now we defined a functor that can handle arbitrary combinations of dense matrix types.
We also specified our preferences how to compute this operation.
When the compiler instantiate our functor for a given type combination it takes
the first product implementation in our list that is admissible.


*/

} // namespace mtl


#endif // MTL_TUTORIAL_INCLUDE
