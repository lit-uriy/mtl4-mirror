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
notation without sacrifying performance.
You can write an expression like x = y * z and the library will
perform the according operation: scaling a %vector, multiplying a
sparse %matrix with a dense %vector or two sparse matrices.
Some operations like dense %matrix product use tuned BLAS implementation.
In parallel, all described operations in this manual are also realized in C++
so that the library can be used without BLAS and is not limited to types
supported by BLAS.
For short, general applicability is combined with maximal available performance.
We developed new techniques to allow for:
- Unrolling of dynamicly sized data with user-define block and tile sizes;
- Combining multiple %vector assignments in a single statement 
  (and more importingly perform them in one single loop);
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
matrices--to save memory bandwidth--while the %vectors are double-valued, 
one cannot use regular BLAS libraries.
In contrast, any generic library that contains a %matrix %vector product
can perform this operation.

And what if somebody wants to build matrices and vectors of quaternions or intervals?
Or rationals?
How to calculate on them?
Again, this is no problem with a generic library but it would take enormous implementation efforts
in Fortran or C (even more in an assembly language to squeaze out the last nano-second of run-time
(on each platform respectively)).


Mathematica and Matlab are by far more elegant than C or Fortran libraries.
And as long as one uses standard operations as %matrix products they are fast
since they can use the tuned libraries.
As soon as you start programming your own computations looping over elements
of the matrices or vectors your performance won't be impressive, to say the least.

MTL4 allows you to write A = B * C and let you use BLAS internally if available.
Otherwise it provides you an implementation in C++ that is also reasonably fast (we usually
reached 60 per cent peak).


All this said, dense %matrix multiplication is certainly the most benchmarked operation
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
This allows for %matrix sizes that are close to the memory size.
It is also possible to change the compressed matrices later.


The product of sparse matrices with dense ones allows you to multiply a sparse %matrix 
simultaneously with multiple vectors.
Besides cache reuse regarding the sparse %matrix simple and efficient loop unrolling
could be applied. (Performance plots still pending ;-) ) 

Sparse matrices can be multiplied very fast with MTL4.
In the typical case that the number of non-zeros per row and per column is 
limited by a constant for any dimension, 
the run-time of the multiplication is linear in the number of rows or columns.
(Remark: we did not use the condition that the number of non-zeros in the %matrix is proportional to 
the dimension. This condition includes the pathological case that the first %matrix contains
a column %vector of non-zeros and the second one a row %vector of non-zeros. Then
the complexity would be quadratic.)
Such matrices usually originate from FEM/FDM/FVM discrezations of PDEs on continous domains.
Then the number of rows and columns corresponds to the number of nodes or cells in the 
discretized domain.
Sparse %matrix products can be very useful in algebraic multigrid methods (AMG).

Returning to the expression A = B * C; it can be used to express every product of 
sparse and dense matrices.
The library will dispatch to the appropriate algorithm.
Moreover, the expression could also represent a %matrix %vector product if A and C
are column vectors (one would probably choose lower-case names though).
In fact,  x = y * z can represent four different operations:
- %matrix product;
- %matrix %vector product;
- scalar times %matrix; or
- scalar times %vector.
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
  - VC 9.0 from Visual Studio 2008

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

P.S.: The description how to use Eclipse is contributed by Michael Schmid
      and we are very grateful for his efforts.
*/






//-----------------------------------------------------------

/*! \page tutorial Tutorial

MTL4 is still in an early state of development and it is possible that
some details may change during further implementation.
However, we will do our best that applications are minimally affected.
In particular, the topics in the tutorial are not subject to modifications.
This, of course, does not exclude backward-compatible extensions.



-# %Vector and %Matrix Types
   -# \subpage vector_def
   -# \subpage matrix_types
   .
-# Generic Insertion
   -# \subpage vector_insertion
   -# \subpage matrix_insertion
   .
-# Assignment
   -# \subpage vector_assignment
   -# \subpage matrix_assignment
   .
-# Operators
   -# \subpage vector_expr 
   -# \subpage rich_vector_expr 
   -# \subpage matrix_expr 
   -# \subpage matrix_vector_expr
   .
-# Norms
   -# \subpage vector_norms 
   -# \subpage matrix_norms 
   .
-# Reductions
   -# \subpage vector_reductions 
   .
-# Other Functions
   -# \subpage conj_intro
   -# \subpage trans_intro
   -# \subpage hermitian_intro
   -# \subpage sub_matrices
   -# \subpage permutation
   -# \subpage banded_matrices
   -# \subpage rank_update
   -# \subpage other_matrix_functions
   .
-# Solving Linear Systems
   -# \subpage trisolve_intro
   -# \subpage krylov_intro
   -# \subpage using_solvers
   .
-# Traversing Matrices and Vectors
   -# \subpage iteration
   -# \subpage rec_intro
   .
-# Advanced Topics
   -# \subpage function_nesting
   .
-# Discussion
   -# \subpage copying
   -# \subpage shallow_copy_problems 
   -# \subpage peak_addiction
-# Performance
   -# \subpage performance_athlon


*/

//-----------------------------------------------------------



/*! \page vector_def Vector Types

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

\if Navigation \endif
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_types 


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

\if Navigation \endif
  Return to \ref vector_def &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_insertion 

*/

//-----------------------------------------------------------




/*! \page vector_insertion Vector Insertion

Vectors are filled by setting the elements, e.g.:
\code
  v[1]= 7.0; v[4]= 8.0;
\endcode
If all elements are equal, one can set it in one statement:
\code
  v= 9.0;
\endcode

\if Navigation \endif
  Return to \ref matrix_types &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_insertion 


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

\section element_insertion Element-wise Insertion

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

\section block_insertion Block-wise Insertion

A more powerful method to fill sparse (and dense) matrices provide the two functions
element_matrix() and element_array().

The following program illustrates how to use them:

\include element_matrix.cpp

The function element_array is designed for element matrices that are stored as 
a 2D C/C++ array.
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

\section init_from_array Initializing Matrices with Arrays

For small matrices in examples it is more convenient to initialize the %matrix from a 2D C/C++ array
instead of filling it element-wise:

\include array_initialization.cpp

C/C++ arrays can be initialized be nested lists.
All MTL4 %matrices provide construction from arrays.
Unfortunately, it is not (yet) possible to initialize user-defined types with lists.
This is proposed for the next C++ standard and we will incorporate this feature 
as soon as it is generally available.

\if Navigation \endif
  Return to \ref vector_insertion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_assignment 

*/

//-----------------------------------------------------------


/*! \page vector_assignment Vector Assignment


Vectors assignments are in most cases performed by expression templated 
(see \ref vector_expr).


Functions that return vectors are subject to move semantics (see \ref copying
and \ref move_semantics).



\if Navigation \endif
  Return to \ref matrix_insertion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_assignment 

*/

//-----------------------------------------------------------


/*! \page matrix_assignment Matrix Assignment

Assignment of matrices verifies the dimensions, i.e.
\code
A= B;
\endcode
is only correct when A and B have the same number of rows and columns.
If you need for some (probably avoidable) reason need to assign matrices of 
different dimension you must explicitly change it:
\code
A.change_dim(num_rows(B), num_cols(B));
A= B;
\endcode
We strongly recommand to avoid this because you risk to hide errors in your program.
Assigning matrices of different dimension is in most cases an indication for an error.
If memory consumption is the reason for such an assignment you should try to destroy unused
matrices (e.g. by introducing additional blocks and define matrices within) and define
new ones.

\section stem_cells Matrix Stem Cells

There is one exception that allows for the change of dimension, when the target has 
dimension 0 by 0.
These matrices are considered as stem cells, they can become whatever desired but 
once they get a non-trivial dimensionality they obey algebraic compatibility rules.
Default constructors of %matrix types always create 0 by 0 matrices.
This simplifies the implementation of generic setter function:
\code
dense2D<double> A;
some_setter(A);
\endcode

\section move_semantics Move Semantics

For numeric reliability we refrain from shallow copy semantics, cf. \ref shallow_copy_problems.
There is an important exception that covers most
algorithmically interesting cases where
shallow copies are legitimate.
Resulting objects of functions and operators exist only once and are
destroyed after assignments.
In the C++ community such arguments that can only appear on the
right-hand side of an assignment are called rvalue.
Rvalues that own their data can be copied shallowly without affecting the semantics.
David Abrahams et al. formalized this approach and implemented
the move library in the 
<a href="http://opensource.adobe.com/group__move__related.html">Adobe Source Libraries (ASL)</a>.

MTL4 uses move semantics to assign matrices of the same type when the source is an rvalue.
Therefore, returning matrices (or vectors) in functions is rather cheap if the target has the same type, e.g.:

\include move_matrix.cpp

Assigning expressions to matrices or vectors does not use move semantics because MTL4 operators are implemented
with expression templates and avoid unnecessary copies with other techniques.
We assume that carefully designed algorithms use assignments of variables to copy their contents and that
after changing one of the two variables the other still have the same value.
\code 
x= y;
y= z;  // x has still the same value as before this operation
\endcode
Resuming this, you can (and should) take an algorithm from a text book, 
implement it with the same operators and functions using MTL4
- Without fearing aliasing effects; and
- Without unnecessary copies.

Please not that move semantics relies on compiler-intern optimizations that some
compilers do not perform without optimization flags, e.g. MSVC.
Therefore, the tests for move semantics are not in the regular test directory
but in another one where the compilation uses optimization flags.
On MSVC we noticed that for higher optimization some locations were equal that
should not.
This could be worked around by inserting print-outs of pointers.
(Nevertheless this is not satisfying and help would be welcome.)

Last but not least, we want to thank David Abrahams and Sean Parent who helped 
to understand the subtle interplay between details of the implementation and
the behavior of the compiler.

\if Navigation \endif
  Return to \ref vector_assignment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_expr 


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
the %vector's value type if necessary to represent the products.
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

\if Navigation \endif
  Return to \ref matrix_assignment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref rich_vector_expr 


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

\if Navigation \endif
  Return to \ref vector_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_expr 


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

Products of three matrices are supported now.
Internally they are realized by binary products creating temporaries
(thus, sequences of two-term products should provide better performance). 
Moreover, products can be arbitrarily added and subtracted:

\include matrix_mult_add.cpp

\if Navigation \endif
  Return to \ref rich_vector_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_vector_expr 

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

%Matrix-vector products (MVP) can be combined with other %vector
operations.
The library now supports expressions like
\code
r= b - A*x.
\endcode

Also supported is scaling of arguments, as well for the %matrix
as for the %vector:

\include scaled_matrix_vector_mult.cpp

All three expressions and the following block
compute the same result.
The first two versions are equivalent: %matrix elements are more numerous
but only used once while %vector elements are less in number but accessed more
often in the operation.
In both cases nnz additional multiplications are performed where nnz is the
number of non-zeros in A.
One can easily see that the third expressions adds 2 nnz operations, 
obviously much less efficient.

Under the assumption that n is smaller than nnz,
clearly less operations are required when the %matrix %vector product is
performed without scaling and the result is scaled afterward.
However, on most computer MVP is memory bandwidth limited and most likely
the additional sweep costs more time than the scaling in the expressions
above.
With the strong bandwidth limitation in MVP, the scaling in the three
expression will not be perceived for most large vectors (it will be done while
waiting anyway for data from memory).


\if Navigation \endif
  Return to \ref matrix_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_norms 


*/

//-----------------------------------------------------------


/*! \page vector_norms Vector Norms


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

\if Navigation \endif
  Return to \ref matrix_vector_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_norms 


*/

//-----------------------------------------------------------


/*! \page matrix_norms Matrix Norms

Norms on matrices can be computed in the same fashion as on vectors:

\include matrix_norms.cpp

The norms are defined as:
- one_norm(A): \f[|A|_1 = \max_j \sum_i |a_{ij}| \f]
- infinity_norm(A): \f[|A|_\infty = \max_i \sum_j |a_{ij}| \f]
- frobenius_norm(A): \f[|A|_F = \sqrt{ \sum_{i,j} |a_{ij}|^2} \f]


\if Navigation \endif
  Return to \ref vector_norms &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_reductions 

*/

//-----------------------------------------------------------


/*! \page vector_reductions Vector Reductions



The sum and the product of all vector's elements can
be computed:

\include vector_reduction.cpp

As %vector reductions base on the same implementation as norms, the
unrolling can be explicitly controlled as shown in the last
command.
The results of these reductions are the value type of the %vector.

\include vector_min_max.cpp

The dot product of two vectors is computed with the function \ref dot:

\include dot.cpp

As the previous computation the evaluation is unrolled, either with
a user-defined parameter or by default eight times.

The result type of \ref dot is of type of the values' product.
If MTL4 is compiled with a concept-compiler, the result type is 
taken from the concept std::Multiple and without concepts
Joel de Guzman's result type deduction from Boost is used.


\if Navigation \endif
  Return to \ref matrix_norms &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref conj_intro 

*/

//-----------------------------------------------------------


/*! \page conj_intro Conjugates

The conjugate of a %vector is computed by:
\code
  conj(v);
\endcode
The %vector \p v is not altered but a immutable view is returned.

In the same manner the conjugate of a %matrix is calculated:

\include matrix_functions2.cpp

This is as well a constant view.

\if Navigation \endif
  Return to \ref vector_reductions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref trans_intro 


*/

//-----------------------------------------------------------


/*! \page trans_intro Transposed

The transposition of %vector is momentarilly not implemented yet.
It will create a row %vector view on a column %vector and vice versa.

Transposing a %matrix can be realized by:

\include matrix_functions2.cpp

The function conj(A) does not change matrices
but they return views on them.
For the sake of reliability, we conserve the const-ness of the referred
matrix.
The transposed of a constant %matrix is itself constant (this feature alone required 
a fair amount of non-trivial meta-programming).
Only when the referred %matrix is mutable the transposed will be:



\if Navigation \endif
  Return to \ref conj_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref hermitian_intro 


*/

//-----------------------------------------------------------


/*! \page hermitian_intro Hermitian

The Hermitians of vectors will be available as soon as transposition is implemented.

Hermitians of matrices are calculated as conjugates of transposed:

\include matrix_functions2.cpp

It returns an immutable view on the %matrix (expression).


\if Navigation \endif
  Return to \ref trans_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref sub_matrices 


*/

//-----------------------------------------------------------


/*! \page sub_matrices Sub-matrices

Sub-matrices also preserve the const attribute of the referred matrices or sub-matrices:

\include matrix_functions3.cpp

Details on the copy behavior of sub-matrices can be found in  section \ref copy_sub_matrix.


\if Navigation \endif
  Return to \ref hermitian_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref permutation 


*/

//-----------------------------------------------------------


/*! \page permutation Permutations and Reordering

The following example shows how to use permutations:

\include permutation.cpp

The function matrix::permutation returns a sparse %matrix computed from a permutation %vector.
The permutation %vector is defined as where entries come from, i.e. v[i] == j means that the 
i-th entry/row/column after the permutation was the j-th entry/row/column before the permutation.
If your %vector is defined in the inverse manner -- i.e. i.e. v[i] == j signifies that the 
i-th entry/row/column before the permutation becomes the j-th entry/row/column after the permutation --
your permutation %matrix is the transposed of what MTL4 computes: P= trans(matrix::permutation(v)).

Reordering is a generalization of permutation.
The entries in the reorder %vector/array are defined in the same fashion as in the permutation %vector.
However, the number of entries is not required to be equal to the set size of projectes indices.
Therefore, the projected %matrix/%vector may have less rows or columns:

\include reorder.cpp

Indices may appear repeatedly in the reorder %vector implying that the respective rows/columns 
appear multiple times in the resulting %matrix/%vector:

\include reorder2.cpp
 



\if Navigation \endif
  Return to \ref sub_matrices &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref banded_matrices 


*/

//-----------------------------------------------------------


/*! \page banded_matrices Banded Matrix View, Upper and Lower Triangular Views

For any %matrix A the upper and the strict upper triangular part can be accessed with the function 
upper and strict_upper:

\include upper.cpp

The functions return views on the arguments. The resulting view can be used in expressions but
this is not recommended in high-performance applications because the lower triangle is still 
traversed while returning zero values.
For the future it is planned to implement traversal of such views more efficiently.

Likewise lower and strict lower triangle matrices are yielded:

\include lower.cpp

In case of sparse matrices the assignment of a lower triangle %matrix leads to an efficient representation
because the entries in the upper part are not explicitly stored as zeros but omitted entirely.

The most general form of views in this section is returned by the function bands
(in fact the others are implemented by it).
It returns bands in terms of half-open intervals of diagonals.
For instance, the two off-diagonal right from the main diagonal are computed by bands(A, 1, 3):

\include bands.cpp

A tri-diagonal %matrix is returned for the band interval [-1, 2) as in the example above.
For performance reasons it is advisable to store the tri-diagonal %matrix in a compressed
format instead of using it directly.

\if Navigation \endif
  Return to \ref permutation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref rank_update 


*/

//-----------------------------------------------------------


/*! \page rank_update Rank-One and Rank-Two Update

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




\if Navigation \endif
  Return to \ref banded_matrices &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref other_matrix_functions 


*/

//-----------------------------------------------------------

/*! 

\if Comment
  \page other_vector_functions Other Vector Functions

  \if NoNavi \endif
  

\endif

*/

//-----------------------------------------------------------

/*! \page other_matrix_functions Other Matrix Functions


For setting up tests quickly, we implemented some convenience functions that initialize matrices:

\include matrix_functions.cpp

Hessian matrices are scaled by a factor, i.e. \ref matrix::hessian_setup(A, alpha) is:
\f[ A= [a_{ij}] = [\alpha * (i + j)] \f]
The funciton is intended for dense matrices.
It works on sparse matrices but it is very expensive for large matrices.

The Laplacian setup \ref matrix::laplacian(A, m, n) 
initializes a matrices with the same values as a finite difference method 
for a Laplace (Poisson) equation on an \f$m\times n\f$ grid.
The matrix size is changed to \f$(m\cdot n)\times (m\cdot n)\f$.
After the setup the diagonal is 4 and four off-diagonals are mostly set to -1, i.e. a simple
five-point-stencil. 
It is intended for sparse matrices but also works on dense ones.



\if Navigation \endif
  Return to \ref rank_update &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref trisolve_intro 


*/

//-----------------------------------------------------------

/*! \page trisolve_intro Triangular Solvers


Linear systems A * x == b are easy to solve if A is an upper/lower triangular matrix.
We provide a generic function to perform this operation:
\code
  x= upper_trisolve(A, b);
\endcode
The %matrix A must be triangular %matrix otherwise the function can throw an exception.
(For dense matrices the lower part is currently ignored but this might change in the future.)
If A has a unit diagonal, the diagonal entries can and must be omitted if the system is solved by:
\code
  x= unit_upper_trisolve(A, b);
\endcode
The implicit diagonal decreases the stress on the memory bandwidth and avoids expensive divisions.

On matrices with non-unit diagonals, the divisions can be circumvented by inverting the diagonal
once with invert_diagonal(A) and then using:
\code
  x= inverse_upper_trisolve(A, b);
\endcode
Especially if A is used as preconditioner of an iterative method, the substitution of divisions by 
multiplications can lead to a significant speed-up.

Likewise, the functions for lower triangular matrices are defined:
\code
  x= lower_trisolve(A, b);
  x= unit_lower_trisolve(A, b);
  x= inverse_lower_trisolve(A, b);
\endcode







\if Navigation \endif
  Return to \ref other_matrix_functions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref krylov_intro 


*/

//-----------------------------------------------------------

/*! \page krylov_intro Introduction Krylov-Subspace Methods

The natural notation in MTL4 allows you to write Krylov-Subspace methods in the same way as in the mathematical
literature.
For instance, consider the conjugate gradient method as it is realized in the ITL version that is in the process of revision:

\include cg.hpp 

If this iterative computation is performed with MTL4 operations on according objects the single statements are evaluated 
with expression templates providing equivalent performance as with algorithm-adapted loops.
For a system with a million unknowns and a five-point-stencil as matrix (explicitly stored) about 10 iterations with a simple
preconditioner
can be performed in a second on a commodity PC.


\if Navigation \endif
  Return to \ref trisolve_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref using_solvers 


*/

//-----------------------------------------------------------

/*! \page using_solvers Using Predefined Linear Solvers

The following program illustrates how to solve a linear system:

\include ilu_0_bicgstab.cpp

Currently two solvers are available:
- Conjugate gradient: itl::cg(A, x, b, P, iter); 
- Bi-Conjugate gradient: itl::bicg(A, x, b, P, iter); 
- Conjugate gradient squared: itl::cgs(A, x, b, P, iter); and
- BiCGStab: itl::bicgstab(A, x, b, P, iter);
- BiCGStab(2): itl::bicgstab_2(A, x, b, P, iter);  (Preconditioning still missing)
.
More solvers will follow.

As preconditioners we provide at the moment:
- Diagonal inversion: itl::pc::diagonal<Matrix>;
- Incomplete LU factorization without fill-in: itl::pc::ilu_0<Matrix>; and
- Incomplete Cholesky factorization without fill-in: itl::pc::ic_0<Matrix>;
.





\if Navigation \endif
  Return to \ref krylov_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref iteration 


*/

//-----------------------------------------------------------

/*! \page iteration Iteration


Iterative traversal of collections is implemented in MTL4 in two ways:
- By iterators and
- By cursors and property maps
.
The latter is more general and allows especially for sparse structures 
a cleaner abstraction.
Initially MTL4 was implemented entirely with this paradigm but it
has shown that algorithms running exclusively on dense structures
are easier to implement in terms of iterators.

All cursors and iterators are handled by:
- The function begin();
- The function end(); and
- The class range_generator.


They are all templated by a tag that determines the form of traversal.
The following tags are currently available for cursors:
- tag::all: iterate over all elements of a collection (or sub-collection);
- tag::nz: iterate over all non-zero elements;
- tag::row: iterate over all rows;
- tag::col: iterate over all columns;
- tag::major: iterate over the major dimension (according to orientation);
- tag::minor: iterate over the minor dimension (according to orientation).
.
For iterators:
- tag::iter::all: iterate over all elements of a collection (or sub-collection);
- tag::iter::nz: iterate over all non-zero elements.
.
And finally for constant iterators:
- tag::const_iter::all: iterate over all elements of a collection (or sub-collection);
- tag::const_iter::nz: iterate over all non-zero elements.
.

Let's consider cursors in more detail.


\section cursor Cursors 

The approach was proposed by David Abrahams in order to separate the
form of traversal from the manner of access.
A cursor is a tool that can be used to visit different objects of a collection.
In an array it can be compared with a position rather than a pointer
because it is not fixed how one accesses the values.
The traversal is essential the same as with iterators, e.g.:
\code
    for (Cursor cursor(begin(x)), cend(end(x)); cursor != cend; ++cursor)
       do_something(cursor);
\endcode
We will come back to the type Cursor later (please be patient).

In order to have more flexibility we templatized the begin and end functions:
\code
    for (Cursor cursor(begin<tag::all>(x)), cend(end<tag::all>(x)); cursor != cend; ++cursor)
       do_something(cursor);
\endcode
This cursor for instance goes over all elements of a %matrix or %vector, including
structural zeros.

\section nested_cursor Nested Cursors 

Several cursors can be used to create other cursors.
This is necessary to traverse multi-dimensional collections like matrices.
In most cases you will use nested cursors via the tags tag::row and tag::col.
The returned cursor can be a certain collection (e.g. a %vector)
or just a place-holder that only contains some index and reference
to a collection but cannot be used directly in operations.
If the type and orientation permits, one can access the elements with
tag::all or tag::nz, e.g.:
\code
    for (Cursor cursor(begin<tag::row>(x)), cend(end<tag::row>(x)); cursor != cend; ++cursor)
       for (ICursor icursor(begin<tag::nz>(cursor)), icend(end<tag::nz>(cursor)); icursor != icend; ++icursor)
           do_something(icursor);
\endcode
Often it is more efficient to adapt an algorithm to the orientation of a %matrix.
Then it is convenient to use tag::major instead of dispatching for row-major and column major matrices:
\code
    for (Cursor cursor(begin<tag::major>(x)), cend(end<tag::major>(x)); cursor != cend; ++cursor)
       for (ICursor icursor(begin<tag::nz>(cursor)), icend(end<tag::nz>(cursor)); icursor != icend; ++icursor)
           do_something(icursor);
\endcode


\section property_maps Property Maps

The concept of property maps has not only the advantage to allow for different
forms of accessibility of values but also to provide different views or details
of this value.
Matrices have four property maps:
- row;
- col; 
- value; and
- const_value.
.
They are all accessed by dereferenced cursors, e.g.
\code
    for (Cursor cursor(begin<tag::nz>(x)), cend(end<tag::nz>(x)); cursor != cend; ++cursor)
	cout << "matrix[" << row(*cursor) << ", " << col(*cursor) << "] = " 
	     << const_value(*cursor) << '\n';
\endcode
Three of the property maps are constant (guess which).
Obviously only value can be changed. The syntax is the following:
\code
    value(*cursor, 7);
\endcode

\section range_generator Range Generator

The type traits traits::range_generator<Tag, Collection>
is used to determine the type of cursor:
\code
    typedef typename traits::range_generator<tag::row, Matrix>::type c_type;
    typedef typename traits::range_generator<tag::nz, c_type>::type  ic_type;

    for (c_type cursor(begin<tag::row>(x)), cend(end<tag::row>(x)); cursor != cend; ++cursor)
       for (ic_type icursor(begin<tag::nz>(cursor)), icend(end<tag::nz>(cursor)); icursor != icend; ++icursor)
           do_something(icursor);
\endcode
As can be seen in the examples, cursors that represents sub-collections (e.g. rows) can
be used as collection type.

\section iterators Iterators

In some contexts, especially with dense data only,
iterators are simpler to use.
With the property map syntax, one cannot apply operators like +=
or a modifying function.
Therefore we provide iterators for dense matrices and vectors.
For sparse matrices there was no use case so far because iterators
do not reveal which %matrix element they are pointing at.

The usage of iterators is very similar to those of cursors:
\code
    for (Iter iter(begin<tag::const_iter::nz>(x)), iend(end<tag::const_iter::nz>(x)); 
         iter != iend; ++iter)
	cout << "matrix value = " << *iter << '\n';
\endcode
In contrast to the previous examples we can only output the value without the indices.
The type of Iter can be determined with range_generator in the same way.

\section nested_iterators Nested Iterators 

Nesting of iterators is also analog to cursors.
However, iterators only exist to access elements not sub-collections.
The nesting is therefore realized by mixing cursors and iterators.
\code
    for (Cursor cursor(begin<tag::major>(x)), cend(end<tag::major>(x)); cursor != cend; ++cursor)
        for (Iter iter(begin<tag::const_iter::nz>(cursor)), iend(end<tag::const_iter::nz>(cursor)); 
             iter != iend; ++iter)
	    cout << "matrix value = " << *iter << '\n';
\endcode
In the example we iterate over the rows by a cursor and then iterate over the elements with
an iterator.


\section range_complexity Advanced topic: Choosing traversal by complexity

Range generators in MTL4 have a notion of complexity.
That is for a given collection and a given form of traversal it can
be said at compile time which complexity this traversal has.

Dense matrices are traversed with linear or cached_linear complexity.
The latter is used for contiguous memory access over strided ones,
which is also linear but considerably slower.
This distinction is mathematically questionable but useful
in practical contexts.

Sparse matrices have linear complexity when traversed along the orientation.
Traversing compressed matrices perpendicular to the orientation 
(e.g. a CRS %matrix column-wise)
has infinite complexity because it is not implemented.
Moreover, the default (non-spezialized) range_generator has infinite
complexity so that it is per se defined for arbitrary collections and tags.
Whether the range generator is actually really implemented can be tested
by comparing the complexity with infinite (by using MPL functions).

The following example shows a simpler way to find out the best traversal:
\include minimize_complexity.cpp

Please not that the example uses compressed sparse matrices and not all
forms of traversion are supported.
Obviously a linear complexity is lower than an infinite and the 
range generator without implemenation is never used.
As the free functions begin() and end() are internally always implemented
by member functions of range_generator (free template functions cannot be 
spezialized partially) we used directly the member functions in the example.


The range generator can also be minimized recursively between three and
more alternatives:
\code
    typedef typename min<range_generator<tag::row, Matrix>, 
	                 typename min<range_generator<tag::col, Matrix>,
	                              range_generator<tag::major, Matrix> >::type 
                        >::type range_type;
\endcode

In many cases there is no need for explicitly minimizing the complexity because
tag::major usually will yield the same results (but this is not so cool).



\if Navigation \endif
  Return to \ref using_solvers &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref rec_intro 


//-----------------------------------------------------------


/*! \page rec_intro Recursion


Recursion is an important theme in MTL4.
Besides matrices with recursive recursive memory layout -- cf. \ref matrix_types and \ref morton_dense --
%recursion with regard to algorithms plays a decisive role.

To support the implementation of recursive algorithms we introduced -- in collaboration with David S. Wise --
the concept to Recursator, an analogon of <a href=" http://www.sgi.com/tech/stl/Iterators.html">Iterator</a>.
The class matrix::recursator enables recursive subdivision of all matrices with a sub_matrix function
(e.g., dense2D and morton_dense).
We refrained from providing the sub_matrix functionality to compressed2D; this would possible but very inefficient
and therefor not particularly useful.
Thus %matrix::recursator of compressed2D cannot be declared.
A recursator for vectors is planned for the future.

Generally spoken, the matrix::recursator (cf. \ref recursion::matrix::recursator)
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


\if Navigation \endif
  Return to \ref iteration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref function_nesting 


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

\if Navigation \endif
  Return to \ref rec_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref copying 


*/

//-----------------------------------------------------------

/*! \page copying Copying in MTL4

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
the other object, see also \ref shallow_copy_problems.

\section copy_sub_matrix Copying Sub-matrices

Sub-matrices are a special case.
The expression
\code 
Matrix E= sub_matrix(A, 2, 5, 1, 9);
\endcode
means that E is defined as a mutable sub-matrix of A.
Internally this is realized as a view on some of A's values.
One could compare this to a window on A. 
As a result, modifications of E affect A and modifications of A change
E if the change was in the range of rows and columns that E refers to.
This  admittedly behaves similarly to shallow copy behavior but is nevertheless
different.
In the case of a sub-matrix, we explicitly request aliasing.
The modification of A can easily prevented by a const argument
\code 
const Matrix E= sub_matrix(A, 2, 5, 1, 9);
\endcode
Furthermore, the sub-matrix of a const %matrix (or another const sub-matrix)
is const itself.
Unless explicitly casted away, const-ness is conserved within MTL4
and cannot be circumvented like in other libraries with shallow copy assignment.
Resuming, the construction of a %matrix with sub_matrix is not a
shallow copy but the definition of a reference to a part of another %matrix.

Once sub-matrix is defined, assignments are regular deep copies, i.e.
\code
E= B;
\endcode
copies the values of B to E and implicitly to the corresponding entries of A.
Sub-matrices are not move semantics, i.e.
\code
E= f(B);
\endcode
cannot use move semantics.
It is correct regarding the destruction of the temporaries and the values of E
but not concerning the modifications of A, which we defined E being a sub-matrix of.

If you do not want the aliasing behavior of sub_matrix but are only interested
in the values of the sub-matrix, you can use the function \ref clone.
\code 
Matrix F= clone(sub_matrix(A, 2, 5, 1, 9));
\endcode
Then deep copy is explicitly used.
F and A are thus entirely decoupled: any modification of either of them
will not affect the other.

Any older remarks on inconsistencies between copy construction and assignment
are invalid now.
In addition, every expression that can be assigned can also be used in copy
constructors, e.g.:
\code
compressed2D<double> A(B * C * D + E);
\endcode


\if Navigation \endif
  Return to \ref function_nesting &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref shallow_copy_problems 


*/

//-----------------------------------------------------------

/*! \page shallow_copy_problems Why Not Using Shallow Copy in Numerical Software

Shallow copy has the advantage over deep copy of being considerably faster.
This advantage does not justify all the dangers implied.

\section scp_unawareness Unawareness

The first risk is that many programmers are not aware of the aliasing
behavior, which is that 
after the assignment neither of the two arguments can be modified without 
affecting the other.
As one of the two variables can be changed in a sub-function of a sub-function of a ...
it is hard to track down all possible modifications.

\section scp_type_dependence Type Dependence of Copy behavior


Moreover, the problem is even more confusing.
Since shallow copy semantic is only feasible between objects of the same type,
assignments between different types must copy data deeply.
In generic functions aiming for maximal generality one do not want assume or
require equality or distinctness of argument types so that the copy behavior 
is unknown.

\include shallow_copy_problems_type.cpp

\section scp_operations Impact of Mathematically Neutral Operations


In the same way mathematically neutral operations like multiplications with one
or additions of zero vectors silently change the program behavior by 
disabling shallow copies and eliminating the aliasing behavior.

\code
A= B;           // Aliasing of A and B
A= 1.0 * B;     // A and B are independent
\endcode

\section scp_obfuscations Code Obfuscation

Many higher level libraries like ITL assigns vectors with the
copy function instead of the assignment operator in order to guarantee deep
copy.

\code 
A= B;           // (Potential) shallow copy
copy(B, A);     // Deep copy
\endcode

We refrain from this approach because this syntax does not correspond to the
mathematical literature and more importantly we cannot be sure that all users
of a library will replace assignments by copy.




\section scp_undermining Undermining const Attributes


Last  but not least
all shallow copy implementations we have seen so far
relentlessly undermined const attributes of arguments.


\include shallow_copy_problems_const.cpp

After calling f, A is modified despite it was passed as const argument and the 
const-ness was not even casted away.

\section scp_resume Resume

For all these reasons we are convinced that reliable mathematical software
can only be implemented with
deep copy semantics.
Unnecessary copies can be avoided by using advanced techniques as expression
templates and \ref move_semantics.

\if Navigation \endif
  Return to \ref copying &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref peak_addiction 

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

Resuming, assembly libraries allow for maximal speed on a rather limited number of types.
Advanced template programming establishes almost competitive performance on an infinite set of types 
while enabling the assembly performance where available.
So, one can write applications with matrices and vectors of genuine or user-defined types
and enjoy maximal available speed.
And we dare to bore the reader with the repetition of the fact that applications only contain
code like A = B * C and the library chooses the optimal implementation.
So, what do you have to loose except that your programs look nicer?


\if Navigation \endif
  Return to \ref shallow_copy_problems &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref performance_athlon 

*/

//-----------------------------------------------------------



/*! \page performance_athlon Performance on an AMD Athlon 2GHz


The following measurements are performed with
 the benchmark template library (BTL) from Laurent Plagne.

The first benchmark is the product of a dense matrix 
with its transposed, i.e. A * trans(A):

\image html athlon/bench-aat.png
\image latex athlon/bench-aat.eps "Matrix product $AA^T$." width=10cm

This operation is favorable for row-major matrices because they are processed with stride 1.
As a consequence the C and C++ implementations perform well for large matrices compared
with the Fortran implementation (where both arguments are traversed with long strides).
The MTL4 implementation is even less affected by the matrix size thanks to the recursive approach.

The implemenation uses tiling on block-level (typically 64 by 64).
For the considered processor a tiling of 2 by 4 yields the performance while processors with more
available FP registers (e.g. PowerPC) are faster with 4 by 4 tiling.
The metaprogramming tuning in MTL4 allows the user to define these parameters in type definitions
of functors and the unrolled implementation is generated at compile time.

In this measurement, the benchmark was compiled without -DMTL_HAS_BLAS (/DMTL_HAS_BLAS on MSVC).
If we had enabled BLAS in MTL4, the two curves would have been identical.

The second example transposes the first argument in the dense matrix product, i.e. trans(A) * A.
This operation is correspondingly more appropriate for column-major matrices so that the
Fortran implementation scales better than the C/C++ codes:

\image html athlon/bench-ata.png
\image latex athlon/bench-ata.eps "Matrix product $A^TA$." width=10cm

As for MTL4, the performance is decreased as well with respect to the first benchmark but
the effect is limited due to the recursive implementation.

Multiplying matrices of the same orientation without transposition, i.e. A * A, scales poorly for 
row-major and column-major if no blocking is used:

\image html athlon/bench-matrix_matrix.png
\image latex athlon/bench-matrix_matrix.eps "Matrix product $AA$." width=10cm

As for the previous measurements, the nested blocking of GotoBLAS and the recursive blocking
of MTL4 cope with the locality problems of large matrices.
In this plot, we also compare with the performance of using recursive matrix formats.
The trend is similar to traditional row-major layout but the performance behaves more stably.
While row-major matrices with strides that are large powers of two introduce a fair amount
of cache conflicts the improved locality of the recursive layout minimizes such conflicts.

The following benchmark considers a different operation, which is 
x= alpha * y + beta * z with alpha and beta scalars and x, y, z vectors.

\image html athlon/bench-vecbinexpr.png
\image latex athlon/bench-vecbinexpr.eps "$x= \alpha y + \beta z$." width=10cm

Most modern libraries use expression templates for this calculation so that all
operations are performed in one single loop.

Finally, we managed outperforming GotoBLAS in one function at least for some sizes:

\image html athlon/bench-dot.png
\image latex athlon/bench-dot.eps "Dot product." width=10cm

The dot product in this plot used internally unrolling with block size 8.
Please note that this is compiler generated code not unrolled by hand.

Thanks to Laurent Plagne for his support with the BTL and to Chris Cole for running the
programs on a cluster node at Indiana University.

\remark
Performance measurings labeled MTL represent computations with MTL2.
MTL2 was tuned for KCC and achieved excellent performance with this compiler (cf. 
<a href="http://osl.iu.edu/research/mtl/performance.php3">MTL2 performance</a>).
With MTL4 we did not rely on compilers for tiling, loop unrolling and similar transformations.
There are two reasons for this: one is that compilers have very different behavior in this regard.
The other reason is that many transformation rely on mathematical properties as commutativity 
that are not known for user types and/or user-defined operations so that compiler optimization is limited
to build-in types and operations.
To cope with this, we implemented accelerating transformation by meta-programming and count
on compilers regarding efficient inlining and reference forwarding.
Our meta-programming optimizations -- short meta-tuning -- proved high efficiency in multiple
measurings (the plots above are only few examples) and were always as fast as code directly 
written in unrolled/tiled form.



\if Navigation \endif
  Return to \ref peak_addiction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 


*/

//-----------------------------------------------------------




// xxxxxxxxxxxxx


/*! \page performance_disclaimer Disclaimer


Unfortunately, the dispatching to BLAS is currently only available for %matrix multiplication.
We work on the extension to other %operations and are not to too proud to accept some generous help.

*/

} // namespace mtl


#endif // MTL_TUTORIAL_INCLUDE
