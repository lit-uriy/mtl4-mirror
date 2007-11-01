// $COPYRIGHT$

#ifndef MTL_TUTORIAL_INCLUDE
#define MTL_TUTORIAL_INCLUDE

// for references
namespace mtl {

// This file contains no source code but only documentation.

/*! \mainpage MTL4 manual

\author Peter Gottschling and Andrew Lumsdaine

Many things can be realized on a computer very elegantly and efficiently today
thanks to progress in software and programming languages.
One thing that cannot be done elegantly on a computer is computing.
At least not computing fast.

In the %Matrix Template Library 4 we aim for a natural mathematical 
notation without sacrifying performancs.
You can write an expression like x = y * z and the library will
perform the according operation: scaling a vector, multiplying a
sparse matrix with a dense vector or two sparse matrices.
Some operations like dense matrix product use tuned BLAS implementation.
In parallel, all described operations in this manual are also realized in C++
so that the library can be used without BLAS and is not limited to types
supported by BLAS.
For short, general applicability is combined with maximal available performance.
We developed new techniques to allow for:
- Unrolling of dynamicly sized data with user-define block and tile sizes;
- Combining multiple vector assignments in a single statement (and more importingly perform them in one single loop);
- Storing matrices recursively in a never-before realized generality;
- Performing operations on recursive and non-recursive matrices recursively;
- Filling compressed sparse matrices efficiently;
.
and much more.

The manual still not covers all features and techniques of the library.
But it should give you enough information to get started.

- \subpage intro 
- \subpage install 
- \subpage IDE
- \subpage tutorial  



*/

//-----------------------------------------------------------

/*! \page intro Introduction




Many things can be realized on a computer very elegantly and efficiently today
thanks to progress in software and programming languages.
One thing that cannot be done elegantly on a computer is computing.
At least not computing fast.

High performance computing (HPC) is to a large extend influenced by some
highly tuned numeric libraries.
Assume we want to multiply two matrices, i.e. calculate A = B * C.
Then we can use some libraries that run at over 90 per cent peak performance.
We only need to write something like:
\code
	int m= num_rows(A), n= num_cols(B), k= num_cols(A), 
            lda= A.get_ldim(), ldb= B.get_ldim(), ldc= C.get_ldim();
	double alpha= 1.0, beta= 1.0;
	char a_trans= 'N', b_trans= 'N';
	_dgemm(&a_trans, &b_trans, &m, &n, &k, &alpha, &A[0][0], &lda, 
	       &B[0][0], &ldb, &beta, &C[0][0], &ldc);
\endcode
No doubt, next time we call dgemm we instantly remember the exact order of the 13 arguments.
Certainly, calling the C-BLAS interface looks somewhat nicer and we can write functions
that deal with the dimensions and the orientation, like dgemm(A, B, C).
We can furthermore write a polymorphic function gemm that accordingly calls _sgemm, _dgemm
and so on.
Indead, there is a project working on this.
But is this all we want?
Why not writing A = B * C; and the library calls the according BLAS function?
What do we want to do if there is none?


Programmers working with BLAS libraries
are forced to limit themselves to the operations and types provided by these
packages.
As an example, if one likes to use single-precision floats for preconditioner
matrices--to save memory bandwidth--while the vectors are double-valued, 
one cannot use regular BLAS libraries.
In contrast, any generic library that contains a matrix vector product
can perform this operation.

And what if somebody wants to build matrices and vectors of quaternions or intervals?
Or rationals?
How to calculate on them?
Again, this is no problem with a generic library but it would take enormous implementation efforts
in Fortran or C (even more in an assembly language to squeaze out the last nano-second of run-time
(on each platform respectively)).


Mathematica and Matlab are by far more elegant than C or Fortran libraries.
And as long as one uses standard operations as matrix products they are fast
since they can use the tuned libraries.
As soon as you start programming your own computations looping over elements
of the matrices or vectors your performance won't be impressive, to say the least.

MTL4 allows you to write A = B * C and let you use BLAS internally if available.
Otherwise it provides you an implementation in C++ that is also reasonably fast (we usually
reached 60 per cent peak).


All this said, dense matrix multiplication is certainly the most benchmarked operation
on high performance computers but not really the operation that high performance computers
use the most in real applications.
The dominant part of scientific computing in HPC are simulations that are mostly 
handled with finite element methods (FEM), finite volume methods (FVM),
finite difference methods (FDM), or alike.
The numeric problems that arise from these methods are almost ever linear or non-linear
systems of equations in terms of very large sparse matrices and dense vectors.

In contrast to most other libraries we paid strong attention to sparse matrices and their
operations.
To start with, we developed an efficient method to fill the matrices and compress them
in-place, cf. \ref matrix_insertion.
This allows for matrix sizes that are close to the memory size.
It is also possible to change the compressed matrices later.


The product of sparse matrices with dense ones allows you to multiply a sparse matrix 
simultaneously with multiple vectors.
Besides cache reuse regarding the sparse matrix simple and efficient loop unrolling
could be applied. (Performance plots still pending ;-) ) 

Sparse matrices can be multiplied very fast with MTL4.
In the typical case that the number of non-zeros per row and per column is 
limited by a constant for any dimension, 
the run-time of the multiplication is linear in the number of rows or columns.
(Remark: we did not use the condition that the number of non-zeros in the matrix is proportional to 
the dimension. This condition includes the pathological case that the first matrix contains
a column vector of non-zeros and the second one a row vector of non-zeros. Then
the complexity would be quadratic.)
Such matrices usually originate from FEM/FDM/FVM discrezations of PDEs on continous domains.
Then the number of rows and columns corresponds to the number of nodes or cells in the 
discretized domain.
Sparse matrix products can be very useful in algebraic multigrid methods (AMG).

Returning to the expression A = B * C; it can be used to express every product of 
sparse and dense matrices.
The library will dispatch to the appropriate algorithm.
Moreover, the expression could also represent a matrix vector product if A and C
are column vectors (one would probably choose lower-case names though).
In fact,  x = y * z can represent four different operations:
- matrix product;
- matrix vector product;
- scalar times matrix; or
- scalar times vector.
.


There is much more to say about MTL.
Some of it you will find in the \ref tutorial, some of it still needs to be written.



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

If you compile MTL4 with VS2005 or its free express version
you need to install the SDK (some boost files access it).
Please make sure that the compiler is in the path.
Then scons will find it.
Additionally, you have to tell the compiler where the header files and
the libraries of VC and the SDK are located, i.e. declare the 
environment variables LIB and INCLUDE. For instance:\n
<tt>LIB=c:/Program Files/Microsoft Visual Studio 8/vc/lib;c:/Program Files/MicrosoftVisual Studio 8/vc/platformsdk/lib</tt>\n
<tt>INCLUDE=c:/Program Files/Microsoft Visual Studio 8/VC/include;c:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include</tt>\n
On some machines the compiler still did not find the files. For that reason the
paths within these two variables are incorporated into the command line by our scons script.



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
For the sake of simplicity, there are no checks in the examples (nevertheless an exceptions
thrown in the examples help to fix a bug).

To compile (and test) all programs you can run scons in the main directory (then you do not
need the -D option and the dot) or in any directory of the tree if you use -D and omit the dot.
You can also compile single files if you specify the name of the executable (including .exe on
windows).

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

\section supported_compilers Supported compilers

The %Matrix Template Library is written in compliance with the C++ standard
and should be compilable with every compiler compliant with the standard.
It has been tested (and passed) with the following compilers and architectures:
- Linux
  - g++ 4.0.1
  - g++ 4.1.1
  - g++ 4.1.2
  - icc 9.0
- Macintosh
  - g++ 4.0.1
- Windows
  - VC 8.0 from Visual Studio 2005

More compilers will be tested in the future.

Compilers that are not standard-compliant (e.g. VC 6.0 from VS 2003) are not subject to support.


Proceed to the \ref IDE.  

*/


//-----------------------------------------------------------


//-----------------------------------------------------------
/*! \page IDE IDE

Some short descriptions how to use MTL4 with different IDE's.

- Eclipse
  - Windows
    - \subpage winxp_eclipse32_gcc323
  - Linux
- MS Visual Studio
  - Visual studio 2005 was successfully used for debugging single files but until now nobody compiled the entire test suite (to our knowledge). 
- WingIDE
  - WingIDE is said to support scons and their is a how-to to this subject. But again, it is not yet tried.
.

Experiences with IDEs are welcome and we would be happy to provide more help in the future.

Proceed to \ref tutorial "the tutorial".  

*/

//-----------------------------------------------------------
/*! \page winxp_eclipse32_gcc323 WinXP / Eclipse-3.2 CDT-3.1 / gcc-3

You should have some basic experience with Eclipse. So I won't explain
each step for downloading and installing Eclipse/CDT. 

Some informations about the used systems:
-# OS: WinXP SP2 with all updates (it's my business notebook, so 
   I can't do something against the updates  :-(   )
-# Compiler: MinGW32 with gcc-3.2.3
-# Eclipse-3.2
-# CDT-3.1.2

Some informations about the installation path:
-# MinGW32: is installed in c:/MinGW
-# Eclipse: is installed in c:/Programme/eclipse
-# CDT-3.1.2: will be installed automatically in the eclipse directory
-# MTL4/Boost: are installed in c:/cppLibs/mtl4 and in c:/cppLibs/boost_1_34_1

Now let's starting Eclipse. If Eclipse is started, change to the c++ perspective.
If this is the first time you can do it under:\n
<tt>Window/Open Persepctive/Other</tt>\n
Now chose \c c++ and the view will get a new look!

To show the configuration we will create a new project. Chose\n
<tt>File/New/Project.../Managed Make C++ Project</tt>\n
This will open a new dialog. Enter <tt>vector1</tt> as project name. I will change
the \c Location to <tt>u:/programming/vector1</tt>. To do this, click on the
check box, now you can push the \c Browse button. The next dialog will open. Chose
a path and in my case, the directory \c vector1 doesn't exist. So I have to
push the button <tt>new directory</tt> and enter the directory name \c vector1.
Now click \c Next.

Click \c Finish on the new dialog. The new project will be created and you can
see it on the left side in the \c Navigator or in the <tt>C/C++ Projects</tt> view.

Now let's copy the \c vector1.cpp of the mtl4 example in the new project directory.
Press \c F5 to update the C++ perspective. Maybe you have to push more than only once.
Java isn't so fast :-)\n
Now you can see the file \c vector1.cpp in the <tt>C/C++ Projects</tt> view.

Before we start with configuring this project, let's check your installation of
MinGW. Enter at the command prompt <tt>gcc --version</tt>. Now something similar
like <tt>gcc (GCC) 3.2.3 (mingw special....)</tt> should appear. Be sure that you 
don't have a second compiler in your path. Please don't install the MSYS package.
This will cause some problems during the linking process. If you get here an error,
please first fix this! Check your path variable and so on. Like the MSYS CYGWIN 
will also cause some problems. Remove the path entry, if you have installed CYGWIN!

Now mark with one left click your project in Eclipse. Than one right click to open 
a context menu. Go down to \c Properties and click again. <tt>Properties for vector1
</tt> dialog appears. Click on <tt>C/C++ Build</tt>. In this section, we will find 
all the necessaries properties we have to configure.

In <tt>Active configuration</tt> you can read \c Debug. For this simple example,
change it to \c Release.

Now in <tt>Configuration Settings / Tool Settings</tt> click on 
<tt>GCC C++ Compiler / Directories</tt>. Here we have to include the
directories of mtl4 and the boost library. We can do it with a click
on the icon with the green cross. In the new dialog, click on 
<tt>File system...</tt> and chose the mtl4 main directory and do the same 
for the boost library. So this property will contain two entries.
-# "C:\cppLibs\mtl4"
-# "C:\cppLibs\boost_1_34_1"
.
\n
in my case.

Now change to the tab <tt>Build Settings</tt>. Enter an artifact name and an
extension. For windows systems this should be \c exe . For artifact name you can
take \c vector1 .\n
Under <tt>Build command</tt> you have to enter <tt>mingw32-make -k</tt>.

So we can go to the next tab \c Environment. I have installed several compiler
vor AVM microcontrollers, CYGWIN and the MinGW. This step is necessary to compile
the example successfull, even though I removed all the compiler entries in the
path variable. Don't ask me why!\n
Click on the button \c New in the configuration section. A next dialog appears.
In the field \c Name enter \c path. In \c Value appears your path and in my
case in the front of all the cygwin installation. Now remove this and all
other compilers in this path (inside the value field). The field \c Delimiter
contains the correct sign. Let's change the \c Operation to \c Replace and
click on OK. So a new user variables appears. Click on apply and than on OK.

Now you can test it if you can compile this simple example. Otherwise, please 
restart Eclipse.

P.S.: The description how to use the Eclipse is contributed by Michael Schmid
      and we are very grateful for his efforts.
*/






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
   .
-# Discussion
   -# \subpage copying
   -# \subpage peak_addiction
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
The %vector in the program above is a column %vector.
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

The following program defines a row %vector of 7 elements without 
(explicit) initialization.

\include vector2.cpp

Scalar values can be assigned to vectors if the type of the scalar
value is assignable to the type of the elements.
Scalar types are in MTL4 all types that are not explicitly defined
by type %traits as vectors or matrices, thus almost all types.
(Unfortunately, we needed to remove the templated assignment on MSVC 8.0
so that only the first assignment with the complex value works there.)

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
%Vector norms are for performance reasons computed with unrolled loops.
Since we do not want to rely on the compilers' capability and 
in order to have more control over the optimization, the unrolling
is realized by meta-programming.
Specializations for certain compilers might be added later
if there is a considerable performance gain over the meta-programming
solution.

Loops in reduction %operations, like norms, are by default unrolled
to 8 statements.
The optimal unrolling depends on several factors, in particular
the number of registers and the value type of the %vector.
The last statement shows how to unroll the computation to six
statements.
The maximum for unrolling is 8 (it might be increased later).

The norms return the magnitude type of the vectors' value type, 
see Magnitude.

Similarly, the sum and the product of all vector's elements can
be computed:

\include vector_reduction.cpp

As %vector reductions base on the same implementation as norms, the
unrolling can be explicitly controlled as shown in the last
command.
The results of these reductions are the value type of the %vector.

In the same way, the maximum and minimum of vectors are computed:

\include vector_min_max.cpp

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
The %vector is not changed but a view on the %vector is created
that conjugate an element when accessed.

The transposition of vectors will be implemented soon.

Return to \ref vector_def "vector definitions"
or proceed to \ref vector_expr "vector expressions".


*/

//-----------------------------------------------------------


/*! \page vector_expr Vector Expressions

The following program illustrates the usage of basic %vector
expressions.

\include vector_expr.cpp

The mathematical definition of %vector spaces requires that
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
and %vector elements are multiplied with the factor when
accessing an element of the view.
Please notice that the scaling factor's type is not required to be
identical with the vector's value type.
Furthermore, the value type of the view can be different from
the vector's value type if necessary to represent the products.
The command is an example for it: multiplying a double %vector
with a complex number requires a complex %vector view to 
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
Expression templates circumvent this repeated loading of %vector
elements by
performing only one loop.


Return to \ref vector_functions "vector functions"
or proceed to \ref rich_vector_expr "rich vector expressions".

*/

//-----------------------------------------------------------


/*! \page rich_vector_expr Rich Vector Expressions

As discussed in the previous chapter, 
%vector operation can be accelerated by improving
their cache locality via expression templates.
Cache locality can be further improved in applications
when subsequent %vector expressions are evaluated
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
assignment with a %vector addition.
The second statement fuses four %vector expressions:
-# The value 2 is assigned to every element of x;
-# w is scaled in-place with 3;
-# v is incremented by the sum of both %vector; and
-# u is incremented by the new value of v.

Again, all these %operations are performed in one loop and each %vector
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
This operation is generic (i.e. applicable to
all %matrix types including sparse).

Just in case you wonder why the %scalar value is only assigned to the diagonal
elements of the %matrix not to all entries, this becomes quite clear
when you think of a %matrix as a linear operator (from one %vector space
to another one).
For instance, consider the multiplication of %vector x with the scalar alpha:
\code
    y= alpha * x;
\endcode
where y is a %vector too.
This %operation is equivalent to assigning alpha to the %matrix A and multiplying x with 
A:
\code
    A= alpha;
    y= A * x;
\endcode
In other words, the %matrix A has the same impact on x as the scalar alpha itself.

Assigning the %scalar value to the diagonal requires of course that the %matrix is 
square.
In the special case that the %scalar value is 0 (more precisely the multiplicative
identity element of the %matrix's value_type) the %matrix can be non-square.
This is consistent with the linear operator characteristic: applying the zero operator
on some %vector results in the zero %vector with the dimension of the operators image.
From a more pragmatic prospective 
\code
    A= 0; 
\endcode
can be used to clear any %matrix, square or rectangular, sparse and dense. 

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
Therefore, the first operation in the function fill is to set the %matrix to zero.


Internally the inserters for dense and sparse matrices are implemented completely
differently but the interface is the same.
Dense inserters insert the value directly and there is not much to say about.

Sparse inserters are more complicated.
The constructor stretches the %matrix so that the first five elements in a row
(in a CCS %matrix likewise the first 5 elements in a column) are inserted directly.
During the live time of the inserter, new elements are written directly into
empty slots. 
If all slots of a row (or column) are filled, new elements are written into an std::map.
During the entire insertion process, no data is shifted.

If an element is inserted twice then the existing element is overwritten, regardless
if the element is stored in the %matrix itself or in the overflow container.
Overwriting is only the default. The function modify() illustrates how to use the inserter
incrementally.
Existing elements are incremented by the new value.
We hope that this ability facilitates the development of FEM code.

For performance reasons it is advisable to customize the number of elements per row (or column),
e.g., ins(m, 13).
Reason being, the overflow container consumes  more memory per element then the 
regular %matrix container.
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
Then the inserter can be used to write values into remote %matrix elements.

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
In the former case the first argument is the element %matrix and the second argument
a %vector containing the indices that correspond to the rows and columns of the
assembled %matrix.
With three arguments, the second one is a %vector of row indices and the third one
a %vector with column indices.
Evidently, the size of the %vector with the row/column indices should be equal to the
number of rows/columns of the element %matrix.

The %vector type must provide a member function size and a bracket operator.
Thus, mtl::dense_vector and std::vector can used (are models).



Return to \ref matrix_types "matrix types"
or proceed to \ref matrix_functions "matrix functions".

*/

//-----------------------------------------------------------


/*! \page matrix_functions Matrix Functions

Norms on matrices can be computed in the same fashion as on vectors:

\include matrix_norms.cpp

Other %matrix functions compute trace of a %matrix,
the element-wise conjugates, and the transposed %matrix:

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

%Matrix multiplication can be implemented as elegantly:


\include matrix_mult_simple.cpp

Arbitrary %matrix types can be multiplied in MTL4.
Although inefficient combinations might be implemented inefficiently--we 
come back to this later.
Let's start with the operation that is the holy grail in 
high-performance computing:
dense %matrix multiplication.
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

Available %matrix-vector %operations are currently multiplication
and updates.
The former can also be expressed by operator and is described
as \ref matrix_vector_expr expression.
The application of rank-one and rank-two updates are
illustrated in the following (hopefully self-explanatory)
program:

\include rank_two_update.cpp

The output of the %matrix is formatted for better readability.
The functions also work for sparse matrices although we
cannot recommend this for the sake of efficiency.

In the future, updates will be also expressible with operators.
For instance, rank_one_update(A, v, w) can be written as
A+= conj(v) * trans(w) if v and w are column vectors (if w
is a row %vector the transposition can-and must-be removed).
Thus, the orientation is relevant in operator notation
where the functions rank_one_update and rank_two_update
ignore the orientation.

Return to \ref matrix_expr "matrix expressions"
or proceed to \ref matrix_vector_expr "matrix-vector expressions".


*/

//-----------------------------------------------------------


/*! \page matrix_vector_expr Matrix-Vector Expressions


%Matrix-vector products are written in the natural way:

\include matrix_vector_mult.cpp

The example shows that sparse and dense matrices can be multiplied
with vectors.
For the sake of performance, the products are implemented with 
different algorithms.
The multiplication of Morton-ordered matrices with vectors is
supported but currently not efficient.

As all products the result of a %matrix-vector multiplication can be 
 -# Directly assigned;
 -# Incrementally assigned; or
 -# Decrementally assigned (not shown in the example).
.
to a %vector variable.

Warning: the %vector argument and the result must be different!
Expressions like v= A*v will throw an exception.
More subtle aliasing, e.g., partial overlap of the %vectors
might not be detected and result in undefined mathematical behavior.

%Matrix-vector products (MVP) cannot be combined with arbitrary %vector
operations.
It is planned for the future to support expressions like
r= b - A*x.

Already supported is scaling of arguments, as well for the %matrix
as for the vector:

\include scaled_matrix_vector_mult.cpp

All three expressions and the following two blocks
compute the same result.
However, since %vector elements are accessed multiple times in an MVP
it is inefficient to scale the %vector
and obviously it is an even bigger waste to  scale both
arguments.

The %matrix scaling on the fly requires the same number of %operations
as an in-place scaling with subsequent MVP (as each %matrix element is
used only once).
Thus, it is advisable to scale the %matrix in the expression instead
of scaling it before the multiplication and scaling it back afterwards
(as the fourth block of %operations).
Scaling the %vector upfront is more efficient under the quite likely
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
%recursion with regard to algorithms plays a decisive role.

To support the implementation of recursive algorithms we introduced -- in collaboration with David S. Wise --
the concept to Recursator, an analogon of <a href=" http://www.sgi.com/tech/stl/Iterators.html">Iterator</a>.
The class matrix_recursator enables recursive subdivision of all matrices with a sub_matrix function
(e.g., dense2D and morton_dense).
We refrained from providing the sub_matrix functionality to compressed2D; this would possible but very inefficient
and therefor not particularly useful.
Thus matrix_recursator of compressed2D cannot be declared.
A recursator for vectors is planned for the future.

Generally spoken, the matrix_recursator (cf. \ref recursion::matrix_recursator)
consistently divides a %matrix into four quadrants 
- north_west;
- north_east;
- south_west; and
- south_east;
.
with the self-evident cartographic meaning (from here on we abreviate %matrix recursator to recursator).
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
because the creation of sub-matrix might be a relatively expensive %operation (e.g., with morton_dense) 
while the creation of a new recursator requires only a few integer %operations.

The recursator uses internally a virtual bound that is a power of 2 and at least as large as
the number of rows and columns.
In the example, the bound is 16 (as shown by the member function bound).
When computing a quadrant the bound is halved and the starting row and column are potentially increased.
For instance, the north_east quadrant is a virtual 8 by 8 %matrix starting at row 0 and column 8.
The sub-matrix referred by the north_east recursator is the intersection of this virtual quadrant with
the original %matrix A, i.e. an 8 by 2 %matrix starting in row 0 and column 8.

More functionality of recursators is shown in the following example:

\include recursator2.cpp

The function is_empty applied on a recursator computes whether the referred sub-matrix is empty,
i.e. the intersection of the virtual quadrant and the original %matrix A is empty.
The sub-matrix itself is not generated since this test can be performed from size and index information.
In the same way, number of rows and columns of the referred sub-matrix can be computed without its creation.

The function is_full() comes in handy in block-recursive algorithms.
Assume we have a base case of 64 by 64, i.e. matrices with at most 64 rows and columns are treated iteratively.
Then it is worthwile to write a blazingly fast iterative implementation  for 64 by 64 matrices,
in other words when the sub-matrix fills the entire virtual quadrant (when bound is 64).
Thus, the function is_full() can be used to dispatch between this optimized code and the (hopefully not
much slower) code for smaller matrices.


Return to \ref iteration "iteration"
or proceed to \ref copying "copying".


*/

//-----------------------------------------------------------


/*! \page copying Copying in MTL4: Shallow versus Deep

Shallow copy -- i.e. copying data types with complex internal structures 
by only copying pointers at the upper level -- allows for very short
run-time since most of the data is not copied physically but only referred
to in the target object.
The draw-back is that changing either of the objects involved in a shallow
copy will alter the other object too.
Especially in complex mathematical applications this often leads to errors
hard to track down.

For that very reason we refrained from shallow copy semantics in assignments,
that is after 
\code 
x= y; 
\endcode one can change x or y without any impact on
the other object.

Some functions in MTL4 -- like sub_matrix -- return matrices or %matrix-like objects.
Applying  deep copy here would cause a serious performance penalty.
For that reason copy constructors use shallow copy.
Thus the code
\code
matrix_type A(10, 10);
A= B;
\endcode
is not equavalent to
\code
matrix_type A= B;
\endcode

With the current implementation we highly advise you to use the assignment
wherever possible.
The copy constructor is too dangerous in its present form!
In addition, several expressions are transformed when used with assignment
but not with copy constructor.
For instance, 
\code
A= B * C;
\endcode
is internally handled as
\code
mult(B, C, A);
\endcode
A similar transformation does not exist for the copy constructor (yet).

We admit that this situation -- the inconsistency between assignment and copy
constructor -- is not satisfying.
In some future release this misbehavior will be terminated by introducing
move semantics that allows to limit shallow copy to cases where it is permissible.
For now we can only recommend you to use assignments.





Return to \ref rec_intro "recursion"
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

Dense %matrix multiplication is the first operation where all the techniques on
this page are applied.
Of course it is planned to extend other %operations in the same manner.



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
This becomes very handy when users define their own (%matrix) types and 
can also provide specialized implementations for certain functions or operators
which are implemented in terms of functors.


\section functor_sec4 Step 4: Reuse of Functors 


Assume we want implement a functor that multiplies matrices using BLAS routines.
We know upfront that only a few type triplets are supported and all other %matrix types
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
same implementation (an explanation will follow) by setting Assign respectively to
assign::assign_sum, assign::plus_sum, and assign::minus_sum.
At this point we focus on the composition.

The duality of fully and partially templated functors simplifies the syntax of composed
functors significantly.
Already the default type of the backup functor can benefit from the shorter syntax
as shown in the example above.


\section functor_avail Available Functors


MTL4 provides several functors for dense %matrix multiplication:
-# Canonical implementation with 3 nested loops and iterators;
-# A corresponding 3-loop implemtation with cursors and property maps;
-# Tiled products for regular matrices using pointers with
   -# With tile size 2 by 2;
   -# With tile size 4 by 4; and 
   -# Costumizable tile size;
   .
-# Recursive %matrix product with costumizable base case (kernel);
-# Platform optimized implementation; and
   -# So far only one implementation from Michael Adams for Opteron
   .
-# BLAS functor calling the corresponding routines.

All these functors have a Backup parameter which is by default set to 
the canonical implementation with iterators.
The two canonical products support all combination of %matrix types
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
This requires also the definition of the Assign parameter because it is
positioned before.
We keep this combined functor type as a type definition and use
it finally in the BLAS functor.
Here we create directly an object of this type which can be later called like a function:

\include nesting/comp_example.hpp

Now we defined a functor that can handle arbitrary combinations of dense %matrix types.
We also specified our preferences how to compute this operation.
When the compiler instantiate our functor for a given type combination it takes
the first product implementation in our list that is admissible.

Return to \ref copying "copying"
or proceed to \ref peak_addiction.

*/

//-----------------------------------------------------------


/*! \page peak_addiction Addicted to peak performance

Sooner or later it comes the day when new software is benchmarked against existing one.
We believe that we achieved very good performance for C++ standards.
But we are still conceivably slower than hand-tuned machine language codes.
We addressed this issue with a similar strategy as Python did.

Python solves the problem of lower performance by not solving it.
Instead, an interface to C/C++ named SWIG was established.
Now people write core components in performance-critical parts with C/C++
and use them in Python.
This way they benefit of the expressiveness of Python with run-time
behavior comparable to C/C++.

Similarly, we stopped trying to reach peak performance at any rate.
Often the medicilously arranged register choreography of some numeric tools
implemented in assembly language cannot be generated by most compiler as efficiently.

In numbers: while many tuned BLAS libraries reach over 90 per cent peak
performance in dense %matrix multiplication, we achieve typically 60 - 70 per cent peak.
This said, we terminated pushing C++ programs further into areas that today's
compilers are not capable to support.

If tuned BLAS libraries reach such high performance--after a lot of hard work though--why
do not use it? 
Following the antic piece of wisdom "If you can't beat them, join them".

So, we internally (we hesitate to say automagically) use the tuned libraries.
That usage remains transparent to the user.
This way we can provide BLAS performance with a more elegant programming style.
(\ref performance_disclaimer)

In addition, our library is not limited to certain types nor to %operations with
arguments of the same type.
We are able to handle mixed %operations, e.g., multiplying float matrices with double vectors.
And of course, we support matrices and vectors of all suitable user and built-in types.
In both cases, we provide decent performance.

Resuming, assembly libraries allow for maximal speed on rather limited types.
Advanced template programming establishes almost competitive performance on an infinite set of types 
while enabling the assembly performance where available.
So, one can write applications with matrices and vectors of genuine or user-defined types
and enjoy maximal available speed.
And we dare to bore the reader with the repetition of the fact that applications only contain
code like A = B * C and the library chooses the optimal implementation.
So, what do you have to loose except that your programs look nicer?

Return to \ref function_nesting.

\page performance_disclaimer Disclaimer


Unfortunately, the dispatching to BLAS is currently only available for %matrix multiplication.
We work on the extension to other %operations and are not to too proud to accept some generous help.

*/

} // namespace mtl


#endif // MTL_TUTORIAL_INCLUDE
