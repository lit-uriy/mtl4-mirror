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
perform the according %operation: scaling a %vector, multiplying a
sparse %matrix with a dense %vector or two sparse matrices.
Some %operations like dense %matrix product use tuned BLAS implementation.
In parallel, all described %operations in this manual are also realized in C++
so that the library can be used without BLAS and is not limited to types
supported by BLAS.
For short, general applicability is combined with maximal available performance.
We developed new techniques to allow for:
- Unrolling of dynamicly sized data with user-define block and tile sizes;
- Combining multiple %vector assignments in a single statement 
  (and more importingly perform them in one single loop);
- Storing matrices recursively in a never-before realized generality;
- Performing %operations on recursive and non-recursive matrices recursively;
- Filling compressed sparse matrices efficiently;
.
and much more.

The manual still not covers all features and techniques of the library.
But it should give you enough information to get started.

- \subpage intro 
- \subpage install 
- \subpage IDE
- \subpage tutorial  
- \subpage overview_ops  
- \subpage faq



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
are forced to limit themselves to the %operations and types provided by these
packages.
As an example, if one likes to use single-precision floats for preconditioner
matrices--to save memory bandwidth--while the %vectors are double-valued, 
one cannot use regular BLAS libraries.
In contrast, any generic library that contains a %matrix %vector product
can perform this %operation.

And what if somebody wants to build matrices and vectors of quaternions or intervals?
Or rationals?
How to calculate on them?
Again, this is no problem with a generic library but it would take enormous implementation efforts
in Fortran or C (even more in an assembly language to squeaze out the last nano-second of run-time
(on each platform respectively)).


Mathematica and Matlab are by far more elegant than C or Fortran libraries.
And as long as one uses standard %operations as %matrix products they are fast
since they can use the tuned libraries.
As soon as you start programming your own computations looping over elements
of the matrices or vectors your performance won't be impressive, to say the least.

MTL4 allows you to write A = B * C and let you use BLAS internally if available.
Otherwise it provides you an implementation in C++ that is also reasonably fast (we usually
reached 60 per cent peak).


All this said, dense %matrix multiplication is certainly the most benchmarked %operation
on high performance computers but not really the %operation that high performance computers
use the most in real applications.
The dominant part of scientific computing in HPC are simulations that are mostly 
handled with finite element methods (FEM), finite volume methods (FVM),
finite difference methods (FDM), or alike.
The numeric problems that arise from these methods are almost ever linear or non-linear
systems of equations in terms of very large sparse matrices and dense vectors.

In contrast to most other libraries we paid strong attention to sparse matrices and their
%operations.
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
In fact,  x = y * z can represent four different %operations:
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

MTL4 is a pure template library and only the presence of the sources
is required.
The only mandatory requirement is the Boost library collection.
There are two ways to install MTL4:
- With a \ref installpackage "package manager" or
- By  \ref installdownload "downloading".
.
We recommand you the first option if you have administrative rights on your computer.
It is much easier and you will be provided with automatic updates.
Furthermore, boost will be installed as prerequisite.

\section installpackage Install with a Package Manager

We provide the MTL4 sources within 4 packages:
- "mtl" contains the headers of MTL4. This package is the only mandatory one for developing scientific applications.
- "mtl-examples" contains documentation of MTL4.
- "mtl-tests" provides our test suite if you like to check whether the library is properly installed and if your platform is well supported.
- "mtl-all" is a meta-package that install the entire MTL4 sources (i.e. the previous 3 packages).

\subsection debian Installation under Debian, Ubuntu and other .deb-based Linux Distributions

-# Open your package source manager (you find this in the menu System -> Administration -> Software Sources).
-# Go to tab "Other Sources", click on "Add" and insert the line:\n
   <tt>deb http://www.simunova.com/debian main/</tt>\n
   then save and close the package source manager. 
-# Open the package manage (you find this in the menu System -> Administration -> Synaptic Package Manager)
   and "Reload".
-# Click on "Quick search" and type "mtl".
-# Mark the MTL4 packages you want and "Apply" for the installation. Accept all dependent packages.

Your package will warn that you are going to install non-authorized packages.
We will provide signatures later. For the time being please accept the installation as it is.

\subsection rpm  Installation under SuSE, Fedora and other .rpm-based Linux Distribution

-# add the url repository \n
   <tt> http://www.simunova.com/rpm </tt>\n
   to your repository list and update.
   - There will be 2 error messages that come from missing signature.
     Please ignore this warning for the moment.
     Later we will add this signatures to avoid these errors.
-# Search the package "mtl" in your package manager
-# Mark the desired packages and accept all dependent packages.

\subsection installwindows Installation under Windows

This is under development and will be provided soon. For the time being please use \ref installdownload "an archive".

\section installdownload Install a Downloaded Archive

To compile an MTL4 application you only need to
-# <a href="www.boost.org/users/download/">Download and install Boost.</a>
-# <a href="www.simunova.com/en/node/145">Download and install MTL4.</a>
-# Include the libraries in the compile command when not installed in the include path, e.g.\n
  <tt>g++ myapp.cpp -o myapp -I/usr/local/include/boost-1.44 -I/usr/local/include/mtl4</tt>
.

<b>Download and install Boost:</b>
One can do this by hand: download it from the 
<a href="http://www.boost.org">Boost web page</a>
and unpack it in an appropriate directory.
If you have administrator rights on the used computer you can put boost in a directory
in the include path, e.g. /usr/local/include.
Then you can omit the compiler flag for including from the boost directory. 
If multiple versions shall be used on your computer you can only put one in the include path
or you need extra tools like softenv or module to deal with your paths.
More convenient is the installation of boost with a packet manager like synaptic.
We use in the development and testing currently versions between 1.38 and 1.44.
Some earlier versions might work as well but not 1.33 or earlier (e.g. type %traits for
std::complex are missing there).
 The parts of boost used in MTL4 do not need
to be compiled but only included (except for the Supercomputing Edition which is
documented seperately).


<b>Download MTL4:</b>





The recommended form of downloading is to use subversion control (under Windows we recommend 
<a href="http://tortoisesvn.tigris.org/">Tortoise</a>) and under Linux it is part of every
recent distribution.
Go to the directory where you like MTL4 to reside and type:\n
<tt>svn checkout https://simunova.zih.tu-dresden.de/svn/mtl4/trunk mtl4</tt>\n
The adventage of version control is that you can update it easily with\n
<tt>svn update</tt>\n
when new features are added or a bug is %fixed (fortunately not needed very often).

If you prefer downloading an archive,
go to the 
<a href="http://www.simunova.com/en/node/145">MTL4 download page</a>
and download the latest archive.
In principle, MTL4 can also be copied in a directory within the standard include
path to omit the compiler flag for its inclusion.



<b>Install MTL4 on Linux:</b>
MTL4 does not need an installation.
You can simply put the files into any directory you like.
In this case you need to pass the directory as flag when you compile your applications, e.g.:\n
<tt>g++ -I/home/joe/mtl4 ...</tt>\n
After extracting, you will have extra prefixes in your file names.
For instance, if you extract an MTL4 archive
in <tt>/home/joe/download</tt> then you need to compile with\n
<tt>g++ -I/home/joe/download/<archivename>/usr/include ...</tt>\n
or you copy the content of <tt><archivename>/usr/include</tt>
into a shorter path

To avoid the -I flag altogether, you can copy the headers into <tt>/usr/include</tt>.
The MTL4 and the boost directory can be mixed in principle since their files are
disjoint (but not all directories).

<b>On Windows:</b>
If you compile MTL4 with VS2005/08/10 or its free express version
you need to install the SDK (some boost files access to it).
Please make sure that the compiler is in the path.
Then cmake (or scons) will find it.

Additionally, you have to tell the compiler where the header files and
the libraries of VC and the Software Developing Kit are located, i.e. declare the 
environment variables LIB and INCLUDE. For instance:\n
<tt>LIB=c:/Program Files/Microsoft Visual Studio 8/vc/lib;c:/Program Files/MicrosoftVisual Studio 8/vc/platformsdk/lib</tt>\n
<tt>INCLUDE=c:/Program Files/Microsoft Visual Studio 8/VC/include;c:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include</tt>\n

To compile MTL4 programs (including the tests), it is advisable to use 
<a href="http://www.cmake.org/">CMake</a>.
Go into the MTL4 main directory and run:\n\n
<tt>cmake .</tt>\n\n
Possibly, CMake will ask you to specify a generator "-G".
After you have run CMake, several project files will appear in this directory.

Click on "ALL_BUILD.vcproj" (or "ALL_BUILD.vcprojx") and Visual Studio will open
with a project folder containing all test and example programs.




\section optionalinstall Optional Installations


<b>Using BLAS:</b>
Dense %matrix multiplication has an acceleration with BLAS (when the types of the %matrix elements allow).
More BLAS usage is currently under development.
To use this acceleration install a well-tuned BLAS (the original Netlib BLAS was even slower than our
implementation when benchmarked it), preferably with a packet manager
and set the macro MTL_HAS_BLAS.
Although one can set the macro somewhere in the program sources it is recommended for better porting
to define it in the compiler, e.g.:\n
<tt>g++ -DMTL_HAS_BLAS -lblas ...</tt>\n
or\n
<tt>cl /DMTL_HAS_BLAS ...</tt>\n
Of course, the library must be linked as well.

<b>Using LAPACK:</b>
LAPACK is currently not supported but will be soon.


<b>Using UMFPACK:</b>
Programs that use UMFPACK must be compiled with MTL_HAS_UMFPACK
and linked with the UMFPACK library plus the libraries UMFPACK
depends on (AMD and UFConfig).

<b>Using Doxygen:</b>
The MTL4 documentation is available online.
If you like to create a copy on your computer, e.g. to read it when offline, you can create it yourself.
Just run <tt>doxygen</tt> in the main directory and you will find the documentation in <tt>libs/numeric/mtl/doc</tt>.
The HTML version is found in <tt>libs/numeric/mtl/doc/html</tt> and a PDF file in <tt>libs/numeric/mtl/doc/pdf</tt>
(not available online).
The revision number in the page footer is not automatically set.
In the main directory is a script <tt>mtl_doxygen</tt> that updates the footer.
Unfortunately, it does not work under Windows.
One can also 
generate man pages by enabling it in the Doxyfile (in MTL4's root directory).
Doxygen can be downloaded <a href="http://www.doxygen.org">here</a>.

\section Testing

To make sure that MTL4 is completely installed and properly working on your platform,
you can run the same tests as we use
in our development.
The whole test suite can be compiled and executed with few commands.
We are currently in the process of transition from one build system to another one.
- Running the tests with scons has currently a bit more functionality (e.g. enabling a higher warning level) but its maintenance will be terminate at some point in the future, see \subpage testing_scons.
- Using cmake one can already build all tests but they need to be started manually. This will be changed soon in a way that cmake-based testing will be at least as powerful as with scons, see \subpage testing_cmake. 
.
Both build systems support Windows and one can use cmake to generate Visual Studion project folder containing all MTL4 tests and tutorial examples.
 
\section install_nutshell In a nutshell


Resuming, for MTL4 you need to:
- Install Boost and include its directory in the compiler flags (unless already in the include path);
- Install MTL4 and include its directory in the compiler flags (unless already in the include path);
- Optionally install cmake or scons (or both);
- Optionally install some or all of the libraries: BLAS, UMFPACK, and LAPACK; and
- Optionally install doxygen.

\section supported_compilers Supported compilers

The %Matrix Template Library is written in compliance with the C++ standard
and should be compilable with every compiler compliant with the standard.
It is regularly tested - <a href="http://www.simunova.com/en/node/33" target="_top">see here
for a list of compilers are tested nightly</a> and 
<a href="http://www.simunova.com/en/node/182" target="_top">here for the test results</a>.

More compilers will be tested in the future.

Compilers that are not standard-compliant (e.g. VC 6.0) are not subject to support.
Visual Studio is considered standard-compliant from VC 7.1 on but we still had trouble to compile MTL4
and even in VC 8.0 we needed a little work-around.


Proceed to the \ref IDE.  

*/
//-----------------------------------------------------------

//-----------------------------------------------------------
/*! 
\page testing_scons Testing with scons





If you want to run the test programs, you need the build system
<a href="http://www.scons.org">scons</a>.
It is easy to install and takes only a few minutes.
The scons-based build of MTL4 uses the environment variables 
<tt>MTL_BOOST_ROOT</tt> to locate the MTL directory
and <tt>BOOST_ROOT</tt> to locate the Boost directory.



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

*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page testing_cmake Testing with CMake


If you want to run the test programs, you will need the the
cross-platform, open-source and build system <a href="http://www.cmake.org/">CMake</a>. This tool is easy
to install, if you have Ubuntu, you can use the Synaptic Package
Manager to install it, or well typing\n\n
<tt>sudo apt-get install cmake</tt>\n\n
on the terminal.

CMake can compile all examples of MTL4 just typing\n\n
<tt>make</tt>\n\n
 in the directory with the examples, but if you want to doing that, you
must generate the makefiles  first.

\section cmake_preps Preparations:


-# Cmake uses one environment variable and that is BOOST_ROOT to locate the Boost directory.
   In bash for example, you can set it with the comand "export", e.g.:\n\n
    <tt>export BOOST_ROOT=/usr/include/boost</tt>\n\n
    Note: you can write that line in your file ~/.bashrc, to have it all the time.
    With csh or tcsh you need accordingly:\n\n
    <tt>setenv BOOST_ROOT/usr/include/boost</tt>\n\n
    which can be put into ~/.cshrc as well.
-# You need a C++ compiler, e.g. g++. On most Linux distributions, this is installed 
   by default. If not you can install it easily with a package manager.
   
\section cmake_run Running CMake

Now you must go to the directory of MTL4 and write on the terminal:\n\n
<tt>cmake .</tt> \n\n
to create all automatic files to compile the examples of
MTL4.

\section cmake_test Building tests and examples

After that, you can go to the directory "libs/numeric/mtl/examples",
to write "make" and all examples will be compiled.




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

MTL4 is becoming rather stable and changes in the interface will be extremely rare.
It goes without saying that we will do our best that applications are minimally affected.
In particular, the topics in the tutorial are not subject to modifications.
This, of course, does not exclude backward-compatible extensions.



-# %Vector and %Matrix Types
   -# \subpage vector_def
   -# \subpage matrix_types
   -# \subpage multivector
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
   -# \subpage namespace_qualification
   -# \subpage copying
   -# \subpage shallow_copy_problems 
   -# \subpage peak_addiction
-# Performance
   -# \subpage performance_athlon
-# \ref overview_ops
-# \ref faq

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

All vectors have free functions for the number of rows and columns
and the  size.

To find out the number of rows use
\code
  unsigned r= num_rows(v);
\endcode
It returns an unsigned integer (more precisely the size_type of the %vector type).
The result is the number of elements for a column %vector and 1 for 
a row %vector.

Likewise the number of columns is given
\code
  unsigned c= num_cols(v);
\endcode
The result is the number of elements for a row %vector and 1 for 
a column %vector.
The size is given by the function size
\code
  unsigned s= size(v);
  assert (s == r * c);
\endcode
and is the number of elements.
This is equal to the product of the numbers of rows and columns.  
These definitions are consistent with the according functions for matrices (\ref matrix_types).



\if Navigation \endif
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref matrix_types 


*/

//-----------------------------------------------------------

/*! \page matrix_types Matrix Types

Right now, MTL4 provides four %matrix types:
- \ref dense2D;
- \ref morton_dense; 
- \ref compressed; and
- multi_vector, see \ref multivector.

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
Diagonal elements are %matrix entries with identical row and column index.
Therefore scalars can also be assigned to non-square matrices.
This %operation is generic (i.e. applicable to
all %matrix types including sparse).

Just in case you wonder why the %scalar value is only assigned to the diagonal
elements of the %matrix not to all entries, this becomes quite clear
when you think of a %matrix as a linear operator (from one %vector space
to another one).
For instance, consider the multiplication of %vector x with the scalar alpha:
\code
    y= alpha * x;
\endcode
where y is a %vector, too.
This %operation is equivalent to assigning alpha to the %matrix A and multiplying x with 
A:
\code
    A= alpha;
    y= A * x;
\endcode
In other words, the %matrix A has the same impact on x as the scalar alpha itself.

If the %matrix is not square, i.e. the linear operator's domain and image have different
dimensions, the equivalence with the scalar multiplication applies accordingly.
In case that the image has a lower dimension, say m, then only the first m entries of the
vector from the domain are scaled with alpha and the others are ignored.
If the image has an higher dimension then the last m-n entries are zero with
n the dimension of the domain.
When you rely on this behavior please check the revision of your MTL4 library:
old versions, i.e. before revision 6843, considered it erroneous to store
 a non-zero scalar to a non-square %matrix.

From a more pragmatic prospective:
\code
    A= 0; 
\endcode
can be used to clear any %matrix, square or rectangular, sparse and dense. 

Dense matrices with a recursively designed memory layout
can be defined with the type \ref morton_dense:

\include morton_dense.cpp

In the pure Morton order format 2 by 2 sub-matrices are stored contiguously in memory.
4 by 4 matrices constitute of 4 2-by-2-matrices and use consecutive memory.
The continuation of this recursive scheme provides square sub-matrices with power of two
sizes that are in contiguous memory and allow for cache-efficient recursive algorithms.
On the other hand, algorithms that are implemented fully recursively create considerable
overhead for function calls.
We therefore recommend using mixed schemes of %recursion and iteration. 
Particularly efficient are algorithms that operate on mid-size blocks, e.g. 64 by 64,
with regular row-major or column-major layout.
MTL4 provides a virtually infinite number of memory layouts for dense matrices
that are specified by a bitmask.
A detailed description and discussion of recursive matrices and algorithm is
provided in 
<a href="http://www.osl.iu.edu/~pgottsch/ics07.pdf">this conference paper</a>.

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

All matrices have free functions for the number of rows and columns
and the %matrix size, which is understood as the product of the former
and not the number of non-zeros.

To find out the number of rows use
\code
  unsigned r= num_rows(A);
\endcode
It returns an unsigned integer (more precisely the size_type of the %matrix type).
Likewise the number of columns is given
\code
  unsigned c= num_cols(A);
\endcode
The %matrix size is given by
\code
  unsigned s= size(A);
  assert (s == r * c);
\endcode
and is defined as product of the numbers of rows and columns.  
These definitions are consistent with the according functions for vectors (\ref vector_def).

How to fill  sparse matrices is shown in section \ref matrix_insertion.

\if Navigation \endif
  Return to \ref vector_def &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref multivector 

*/

//-----------------------------------------------------------

/*! \page multivector Type Multivector


A multi-vector is an abstraction that is very useful in Krylov subspace methods, esp.
when they are implemented generically.
It is a %matrix that composed of column vectors.
The simplest use case is to just store multiple vectors of the same length.
Especially the %matrix %vector product including its transposed form allow for well-readable
implementations of algorithms like GMRES and for straight-forward extension to distributed
vectors.

To create a multi-vector, there are two ways:
-# Constructor by number of rows and columns: 
\code
	mtl::multi_vector<Vector> 	A(2, 3);
\endcode
-# Constructor by number of rows and column %vector for initialization
\code
	mtl::multi_vector<Vector>	A(Vector(3), 2);
\endcode
In the first method, you get a %matrix which has 2 rows and 3 columns.
This constructor with number of rows and columns exist for all %matrix types and is important for 
generic creation.

In the 2nd method, you get a %matrix with 2 rows and 3 columns.
Remark: in older versions, before revision 6957, the arguments were reversed.

To find out the number of rows use
\code
  unsigned r= num_rows(A);
\endcode
It returns an unsigned integer (more precisely the size_type of the %vector type).
Likewise the number of columns is given
\code
  unsigned c= num_cols(A);
\endcode

To modify individual entries of the multi-vector, there are several possibilities. 
You can write a %vector (same length) in the k-th column of the multi-vector, and vice versa.
\code
	mtl::multi_vector<Vector> 	A(2, 3);
	Vector				v(2), w(2);
	A.vector(k)= v;
	v= A.vector(k);
\endcode
You can also specify the row and column of the entry to be changed.
\code
	A[1][1]= 3.5;
\endcode

\section ops_multi_vector Operations with multi-vectors

On the one hand we can consider a multi-vector as a collection of vectors, on the other hand as a %matrix.
Thus, there are many multi-vector %operations.
\include multi_vector.cpp

The multi-vector is a %matrix, therefore the following %matrix 
%operations are defined: trace(A), conj(A), trans(A), hermitian(A).
The interface is nevertheless minimalistic and not the same functionality as for other %matrix types is provided.
More functions will be implemented when needed.



\if Navigation \endif
  Return to \ref matrix_types &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref vector_insertion 

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

\section destroy_inserter INSERTERS MUST BE DESTROYED

A mistake that many people did with inserters was using the matrix before the inserter
was destroyed, e.g.:
\code
using namespace mtl;
typedef compressed2D<double> matrix_type;

matrix_type A(5, 5);
matrix::inserter<matrix_type> ins(A);
ins[0][0] << 7.3; // .... more insertions

do_something_with(A);  // TROUBLE!!!
\endcode
Then the matrix A is not ready and an exception is thrown (or an assertion fails depending on
compile flags).

The issue was apparently not sufficiently discussed in the tutorial and 
we have to blamed not the users for doing this wrong.


The insertion problem is circumvented by defining the inserter in a separate function
as we did in the \ref element_insertion "previous section".
If we accessed the matrix within the fill-in function 
we would experience the same problem.

If no separate function shall be defined for brevity, one can define the
inserter in an extra block.
The following program implements the function "fill" of the example insert.cpp
with a compressed matrix:

\include insert_scope.cpp

Alternatively, one can handle the insertion destruction explicitly with pointer
as will be explained \ref multiple_insertion "later".

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

\section multiple_insertion Insertion in Multiple Function calls

If a matrix is set up by means of multiple function calls as it happens often
in finite element assembly.
Say we have a class world_matrix that contains a compressed matrix which is
set by calling add_entry several times.
We can define an inserter in the function add_entry and the inserter will be
destroyed after leaving the function:

\include insert_class_expensive.cpp

This works correctly but it is horribly slow because every value inserted need the 
creation of an inserter which is extremely expensive.

Defining an inserter as member of the class does not work at all because
the inserter will live as long as the containing object and the matrix
cannot be accessed during its life time.

The solution is to define a <b>pointer to an inserter as member</b>.
The pointer lives as long as the object and the life time of the referred inserter
can be controlled manually with new and delete.
This said, we need a function that allocates the pointer that must be called
before starting the insertion.
Accordingly, a function is required that deallocates the pointer so that
the inserter is destroyed.
This function must be called after terminating the insertion phase and before
accessing the matrix:

\include insert_class.cpp

Note that "ins" must be dereferred in add_entry.
We find this approach more error-prone than defining the inserter in an
extra scope but im some situations this is the only feasible way.

This approach is used in several software projects such as
<a href="http://www.simunova.com/node/17" target="new">AMDiS</a>
and
<a href="http://www.fenicsproject.org/" target="new">FEniCS</a>.

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


/*! \page permutation Permutations, Reordering, and Matrix Indirection

\section Permutation

The following example shows how to use permutations:

\include permutation.cpp

The function matrix::permutation returns a sparse %matrix computed from a permutation %vector.
The permutation %vector is defined as where entries come from, i.e. v[i] == j means that the 
i-th entry/row/column after the permutation was the j-th entry/row/column before the permutation.
If your %vector is defined in the inverse manner -- i.e. i.e. v[i] == j signifies that the 
i-th entry/row/column before the permutation becomes the j-th entry/row/column after the permutation --
your permutation %matrix is the transposed of what MTL4 computes: P= trans(matrix::permutation(v)).

\section Reordering

Reordering is a generalization of permutation.
The entries in the reorder %vector/array are defined in the same fashion as in the permutation %vector.
However, the number of entries is not required to be equal to the set size of projectes indices.
Therefore, the projected %matrix/%vector may have less rows or columns:

\include reorder.cpp

Indices may appear repeatedly in the reorder %vector implying that the respective rows/columns 
appear multiple times in the resulting %matrix/%vector:

\include reorder2.cpp
 
Reordering matrices can also be used to compress matrices as in the following example:

\include reorder3.cpp

The multiplication from right allows for eliminating empty rows
and from left with the transposed for removing empty columns.
The transposed of the compression matrices enable the decompression to
yield the original matrices.
If memory is an issue, one can also keep the compression vectors and the size of the original %matrix and
create the compression %matrix on the fly.

\code
    dense2D<double>  C1(trans(matrix::reorder(non_zero_rows, 4)) * B3),
                     C2(C1 * matrix::reorder(non_zero_columns, 3));
\endcode


For the definition of the row compression %matrix we needed the explicit specification of the number of columns
because the reorder %matrix has by default the maximal entry plus one as column number
(i.e. the minimum that is necessary).
The number of columns of any reorder %matrix -- not only compression -- must be equal the number of rows
of the original %matrix when multiplied from left and equal the original number of columns when multiplied
from right as transposed.
This is implicitly given when the last row or column is part of the resulting %matrix.
If you are not sure about this fact or the compression %vector is calculated specify the reorder %matrix' columnn number explicitly.

\section Indirection

Matrix indirection, implemented in matrix::indirect, is a view on an existing %matrix  restricted to certain indices.
It uses the type iset to define index sets.
The following program illustrates the usage:

\includelineno matrix_indirect.cpp

An \ref iset can be initialized with push_back() or simply assigned with a comma-separated list (line 13).
 (And yes, the comma operator is overloaded).
When isets are passed as indices to a %matrix, an object of type matrix::indirect is created (line 16).
For the moment, \ref iset cannot be mixed with other types as index (e.g. to get a sub-vector within a matrix):
\code
   A[rows][1]; // Error !!!
\endcode
When the sub-matrix is accessed multiple times, it should be stored into an object as in in line 18.

Objects of type matrix::indirect can be used in operations and this is tested to some extend.
In the future, %matrix indirections from dense matrices (in general, all matrices modifiable without inserter)
can be modified.


\section Comparison

For small sub-matrices, %matrix indirection is always faster than reordering because the latter involves
one or two matrix products.
For large permuted or reordered matrices, the multiplication(s) can be amortized by the  faster access to the elements later.



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

Currently the folling solvers (in alphabetical order) are available:
- Bi-Conjugate Gradient: itl::bicg(A, x, b, L, iter); 
- Bi-Conjugate Gradient Stabilized: itl::bicgstab(A, x, b, L, iter);
- Bi-Conjugate Gradient Stabilized(2): itl::bicgstab_2(A, x, b, L, iter); 
- Bi-Conjugate Gradient Stabilized(ell): itl::bicgstab_ell(A, x, b, L, R, iter, ell); 
- Conjugate Gradient: itl::cg(A, x, b, L, iter); 
- Conjugate Gradient Squared: itl::cgs(A, x, b, L, iter); 
- Generalized Minimal Residual method (without restart): itl::gmres_full(A, x, b, L, R, iter, kmax_in); 
- Generalized Minimal Residual method with restart: itl::gmres(A, x, b, L, R, iter, restart); 
- Induced Dimension Reduction on s dimensions (IDR(s)): itl::idr_s(A, x, b, L, R, iter, s); 
- Quasi-minimal residual: itl::qmr(A, x, b, L, R, iter); and
- Transposed-free Quasi-minimal residual: itl::tfqmr(A, x, b, L, R, iter).
.
All Krylov sub-space methods solve the linear system Ax = b as in the example above.
A left preconditioner  L is used in all methods and some methods also 
incorporate a right preconditioner  R.
The iteration object controls the termination of the iteration, see below.
Some algorithms take an additional argument (ell, kmax_in, restart, s) specifying the dimension of the 
%vector space regarding to which new search directions are orthogonalized.
The difference between gmres and gmres_full is that continues until one criterion in iter holds.
gmres_full computes at most kmax_in iterations (or size(x) depending on what is smaller) 
regardless on whether the termination criterion is reached or not.
Needless to say that gmres is implemented by means of gmres_full.

As preconditioners we provide at the moment:
- Diagonal inversion: itl::pc::diagonal<Matrix>;
- Incomplete LU factorization without fill-in: itl::pc::ilu_0<Matrix>; and
- Incomplete Cholesky factorization without fill-in: itl::pc::ic_0<Matrix>;
.

The iteration object can be chosen between:
- Basic iteration does not generate output: basic_iteration(r0, m, r, a= 0);
- Cyclic iteration prints residual information every  c iteration: cyclic_iteration(r0, m, r, a= 0, c= 100, out= std::cout); and
- Noisy iteration prints residual in each iteration: noisy_iteration(r0, m, r, a= 0, out= std::cout).
.
Mandatory arguments for the iteration objects' constructors are the initial residuum  r0 (which is of course  b if one starts
with  x = 0), the maximal number of iteration and the relative error reduction (more precisely residuum reduction).
Optionally, the absolute residuum  a (fourth argument) can be given as termination criterion.
Thus the iterative methods are terminated when either:
- The maximum number of iterations is reached (failure);
- The relative residuum reduction was achieved; or
- The absolute residuum is below  a if specified.
.
In the cyclic iteration, the user can specify after how many iterations the residual informations are printed,
default is 100. 
For cyclic and noisy iterations, one can declare on which ostream the information is printed.
This enables printing it into log files or for parallel computing printing only on one processor.
By default the output is printed into std::out.

General assumptions on solver iterations:
- 0th iteration is the starting residue.
- Once the input value (x) is changed you have made at least one iteration.
- Fractions of iterations are counted as whole iterations.



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
because it is not %fixed how one accesses the values.
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

The type %traits traits::range_generator<Tag, Collection>
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
  Return to \ref rec_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref namespace_qualification 


*/

//-----------------------------------------------------------

/*! \page namespace_qualification Namespace qualification

All classes and functions are defined in %namespace mtl or sub-namespaces thereof.
Matrix-related functions and types are defined in mtl::matrix and vector material likewise
in mtl::vector.
To make applications shorter, often used types like compressed2D and dense_vector are imported into
the namespace mtl.
As a consequence, you can write mtl::compressed2D<..> or mtl::matrix::compressed2D<..>.

Functions are defined as much as possible in the namespaces mtl::matrix and mtl::vector.
Therefore, Argument-Dependent Lookup (ADL) finds these functions without namespace qualification.
For instance, if we call trans(x) the mtl::matrix::trans() is called if x is a %matrix, i.e.
the type of x is defined in mtl::matrix.
Likewise if x is a %vector.
If the type of x is not defined in MTL4, you must qualify the function because ADL does not apply.
Alternatively, you can import your type into the appropriate namespace:
\code
namespace mtl { namespace matrix {
    using my_namespace::my_matrix_type;
}}
\endcode
In both cases you would also need a handful of type %traits (the documentation of integrating external types
into MTL4 is pending).

To avoid all namespace qualifications, you can use 
\code
using namespace mtl;
\endcode
in your program before you use the first MTL4 item.
For small programs this is probably the easiest solution.

If you work at a larger project, it is better not importing the entire namespace.
Instead you have to qualify all types and NOT qualify the functions.
If you call mtl::trans it will not find it because it is defined in sub-namespaces.
In general, you should avoid explicit qualification as mtl::f(...).

Functions with
a special behavior are those which are defined upon MTL4 types and non-MTL4 types at the same type.
An example is size(). Like trans it is defined in  mtl::matrix and mtl::vector.
It also works for std::vector and for arrays.
The definitions of these functions are in namespace mtl (we did not like defining it in std:: or the global namespace).
The best way using such functions is
\code
using mtl::size;
unsigned n= size(x);
\endcode
This works for all supported types of x.
If x is a matrix then mtl::matrix::size is called and if x is a std::vector mtl::size is called (which is
implemented with partially specialized functor but this is another topic).

As a rule of thumb. If you call an unqualified function for an MTL4 type ADL will find it (otherwise it is sloppily implemented
and must be %fixed).
If you call an MTL4 function for a non-MTL4 type write using mtl::f before calling f.
For generic functions that handle both MTL4 and non-MTL4 types also write using mtl::f.
If it still not compile, the function is probably not implemented yet.



\if Navigation \endif
 Return to \ref function_nesting &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref copying 


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
  Return to \ref namespace_qualification &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref shallow_copy_problems 


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



/*! \page performance_athlon Performance on an AMD Opteron 2GHz


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
  Return to \ref peak_addiction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref overview_ops


*/

//-----------------------------------------------------------
/*! \page overview_ops Overview

-# %Matrix Operations
   - \subpage mat_vec_expr
   - \subpage adjoint
   - \subpage bands
   - \subpage change_dim
   - \subpage conj
   - \subpage crop
   - \subpage diagonal_setup
   - \subpage eigenvalue_symmetric
   - \subpage extract_hessenberg
   - \subpage extract_householder_hessenberg
   - \subpage frobenius_norm
   - \subpage hermitian
   - \subpage hessenberg
   - \subpage hessenberg_factors
   - \subpage hessenberg_q
   - \subpage hessian_setup
   - \subpage householder_hessenberg
   - \subpage infinity_norm
   - \subpage inv
   - \subpage inv_lower
   - \subpage inv_upper
   - \subpage invert_diagonal
   - \subpage laplacian_setup
   - \subpage lower
   - \subpage lu
   - \subpage lu_p
   - \subpage lu_adjoint_apply
   - \subpage lu_adjoint_solve
   - \subpage lu_apply
   - \subpage lu_f
   - \subpage lu_solve
   - \subpage lu_solve_straight
   - \subpage max_abs_pos
   - \subpage max_pos 
   - \subpage nnz
   - \subpage num_cols
   - \subpage num_rows
   - \subpage one_norm
   - \subpage op_matrix_equal
   - \subpage op_matrix_add_equal
   - \subpage op_matrix_add
   - \subpage op_matrix_min_equal
   - \subpage op_matrix_min
   - \subpage op_matrix_mult_equal
   - \subpage op_matrix_mult
   - \subpage qr_algo
   - \subpage qr_sym_imp
   - \subpage rank_one_update
   - \subpage rank_two_update
   - \subpage RowInMatrix
   - \subpage set_to_zero
   - \subpage strict_lower
   - \subpage strict_upper
   - \subpage sub_matrix
   - \subpage svd_tol
   - \subpage svd
   - \subpage swap_row
   - \subpage trace
   - \subpage trans
   - \subpage tril
   - \subpage triu
   - \subpage upper
   .
-# %Vector Operations
   - \subpage dot_v
   - \subpage dot_real_v
   - \subpage infinity_norm_v
   - \subpage iota_v
   - \subpage iota_v1
   - \subpage max_abs_pos_v
   - \subpage max_pos_v
   - \subpage max_v
   - \subpage min_pos_v
   - \subpage min_v
   - \subpage one_norm_v
   - \subpage op_vector_add_equal
   - \subpage op_vector_add
   - \subpage op_vector_min_equal
   - \subpage op_vector_min
   - \subpage orth_v
   - \subpage orth_vi
   - \subpage orthogonalize_factors_v
   - \subpage product_v
   - \subpage size_v
   - \subpage sum_v
   - \subpage swap_row_v
   - \subpage trans_v
   - \subpage two_norm_v
   .
-# %Matrix - %Vector Operations
   - \subpage inverse_lower_trisolve
   - \subpage inverse_upper_trisolve
   - \subpage matrix_vector
   - \subpage lower_trisolve
   - \subpage permutation_av
   - \subpage reorder_av
   - \subpage upper_trisolve
   - \subpage unit_lower_trisolve
   - \subpage unit_upper_trisolve
-# %Scalar - %Vector Operations
   - \subpage scalar_vector_mult_equal
   - \subpage scalar_vector_div_equal
-# miscellaneous
   - \subpage iall
   - \subpage imax
   - \subpage irange

\if Navigation \endif
  Return to \ref performance_athlon &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page mat_vec_expr  A[range1][range2]

returns a submatrix of %matrix A with rows in range1 and cols in range2.

For example, if range1 is a number, the returntype is a row-vector.
\code
using mtl::iall;
dense_vector<cdouble, vector::parameters<tag::row_major> > v_r(A[0][iall]);
\endcode

If the range2 is a number, the returntype is a col-vector.
\code
using mtl::iall;
dense_vector<cdouble>   v_r= dense_vector<cdouble>(A[iall][0]);
\endcode

If range1 and range2 are numbers, the returntype is a scalar.
\code
cdouble C= A[0][0];
\endcode

If range1 and range2 are regions, the resulttype is a %matrix.
\code
irange row(2, 4), col(1, 7);
dense2D<cdouble> B= A[row][col];
\endcode

\include matrix_functions3.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page adjoint  adjoint(A)

Adjoint matrix of an m-by-n matrix A with complex entries is the n-by-m matrix A* obtained from A by taking the transpose and then taking the complex conjugate of each entry (i.e. negating their imaginary parts but not their real parts).


\code
B= adjoint(A);
\endcode

Details:: mtl::matrix::adjoint

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page bands bands(A, begin, end)

Returns a view of a matrix \p A from diagonal \p begin to \p end

The main diagonal is numbered 0; the off-diagonal below the main one is -1.
    Accordingly, the off-diagonal above the main is 1.
    The parameters \p begin and \p end specify a right-open interval.
    For, instance bands(A, -1, 2) yields a tridiagonal matrix as in the code example.

\code
B= bands(A, -1, 2);
\endcode

Details:: mtl::matrix::bands

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page change_dim  change_dim(A)

Ecplicity change of the %matrix dimension.

\code
A.change_dim(num_cols(B), num_rows(B));
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page conj  conj(A)

The conjugate of a %matrix is computed by: 
\code
conj(A);
\endcode
The %matrix A is not altered but a immutable view is returned.

Details: mtl::matrix::conj

\include matrix_functions2.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page crop  crop(A)

Remove all zero entries from a collection. 

Details: mtl::matrix::crop

\code
crop(A);
\endcode

Only for sparse matrices useful.

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page diagonal  diagonal(A)

Returns the %vector with the diagonal of the %matrix A. 

Details: mtl::matrix::diagonal

\code
diagonal(A);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page diagonal_setup  diagonal_setup(A, value)

Setup a %matrix to a multiple of the unity %matrix. (works for all %matrix types in mtl4)

Details: mtl::matrix::diagonal_setup

\code
diagonal_setup(A,2.0);
\endcode

\include setups_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page eigenvalue_symmetric  eigenvalue_symmetric(A)

Returns eigenvalues of symmetric %matrix A.

Currently there are 2 algorithms that compute the eigenvalues of a symmetric %matrix.
1. qr_algo(A, iterations)
2. qr_sym_imp(A) (with Wilkinson shift) 

Details: mtl::matrix::eigenvalue_symmetric

For example:

\include eigenvalue_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page extract_hessenberg  extract_hessenberg(A)

Returns Extracted Hessenberg form from factorization H of some A.

Details: mtl::matrix::extract_hessenberg

\code
E= extract_hessenberg(A);
\endcode

For example:

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page extract_householder_hessenberg  extract_householder_hessenberg(A)

Returns the Householder %vectors from Hessenberg factorization H of some A which are stored in tril(A,-2).

Details: mtl::matrix::extract_householder_hessenberg

\code
B= hessenberg_factors(A);
C= extract_householder_hessenberg(B);
\endcode

For example:

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page frobenius_norm  frobenius_norm(A)

return Frobenius-Norm of Matrix A.

Details: mtl::matrix::frobenius_norm

For example:

\include matrix_norms.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page hermitian  hermitian(A)

The hermitian of a %matrix is computed by: 
\code
hermitian(A);
\endcode

Details: mtl::matrix::hermitian

\include matrix_functions2.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page hessenberg  hessenberg(A)

Returns Hessenberg-Form of %matrix A. (triu(A,-2)).
Hessenberg-Form: upper triangle-matrix and first diagonal under the main-diagonal.

Details: mtl::matrix::hessenberg

\code
B= hessenberg(A);
\endcode

For example:

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page hessenberg_factors  hessenberg_factors(A)

Returns Hessenberg-Form of %matrix A with Householder-vectors in the lower triangle(tril(A,-1)).
The triu(result,-2) is the Hessenberg-Form of %matrix A.

Details: mtl::matrix::hessenberg_factors

\code
B= hessenberg_factors(A);
\endcode

For example:

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page hessenberg_q  hessenberg_q(A)

Returns Q where \f$ Q'*A*Q \f$ == hessenberg(A).

Details: mtl::matrix::hessenberg_q

For example:

\code
F= hessenberg_q(A);
\endcode

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page hessian_setup  hessian_setup(A, value)

Fills a matrix A with \f$ a_{ij} = factor * (i + j) \f$.
Works only for Morton Z-order and Hybrid 2 row-major matrices.

Details: mtl::matrix::hessian_setup

\code
hessian_setup(A,2.0);
\endcode

\include setups_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page householder_hessenberg  householder_hessenberg(A)

Returns the Householder %vectors from Hessenberg factorization H of some A which are stored in tril(A,-2).

Details: mtl::matrix::householder_hessenberg

\code
D= householder_hessenberg(A);
\endcode
same as:
\code
B= hessenberg_factors(A);
C= extract_householder_hessenberg(B);
\endcode

For example:

\include hessenberg_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page infinity_norm  infinity_norm(A)

Returns infinity norm of %matrix A.

Details: mtl::matrix::infinity_norm

For example:

\include matrix_norms.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page inv inv(A)

Returns inverse of %matrix A.

Details: mtl::matrix::inv

\include inv_matrix.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page inv_lower inv_lower(A)

Invert lower triangular %matrix A.
Returns invert %matrix.

Details: mtl::matrix::inv_lower

\include inv_matrix.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page inv_upper inv_upper(A)

Invert upper triangular %matrix A.
Returns invert %matrix.

Details: mtl::matrix::inv_upper

\include inv_matrix.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page invert_diagonal  invert_diagonal(A)

Returns %matrix A with invert diagonal.

Details: mtl::matrix::invert_diagonal


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page laplacian_setup  laplacian_setup(A, dim1, dim2)

return n by n-Laplace %matrix with \f$ n= dim1 * dim2 \f$. (5 Point stencil)

Details: mtl::matrix::laplacian_setup

For example:

\include setups_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page lower  lower(A)

returns lower triangular %matrix.

Details: mtl::matrix::lower

\code
B= lower(A);
\endcode


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page lu lu(A)

without pivoting
return LU-Form of %matrix A and saves in %matrix A. With U as the upper triangular %matrix and L as the lower triangular %matrix. Attention without optimization and pivoting.

Details: mtl::matrix::lu

\code
lu(A);
L= strict_lower(A);  U= upper(A);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page lu_p lu(A,Permutation)

with pivoting
return LU-Form of %matrix A and saves in %matrix A. With U as the upper triangular %matrix and L as the lower triangular %matrix.

Details: mtl::matrix::lu

\code
lu(A,Permutation);
L= strict_lower(A);  U= upper(A);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page lu_adjoint_apply lu_adjoint_apply(A, Permutation, b)


Apply the factorization L*U with permutation P on vector b to solve adjoint(A)x = b.
That is \f$ P^{-1}LU^H x = b \f$ --> \f$ x= P^{-1}L^{-H} U^{-H} b \f$ where \f$ {{P^{-1}}^{-1}}^H = P^{-1} \f$.


Details: mtl::matrix::lu_adjoint_apply

\code
x= lu_adjoint_apply(A, PermVector, b);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page lu_adjoint_solve lu_adjoint_solve(A, b)


Solve adjoint(A)x = b by LU factorization with column pivoting; %vector x is returned.

Details: mtl::matrix::lu_adjoint_solve

\code
x= lu_adjoint_solve(A, b);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page lu_apply lu_apply(A, Permutation, b)


Apply the factorization \f$ L*U \f$ with permutation P on %vector b to solve \f$ Ax = b \f$.

Details: mtl::matrix::lu_apply

\code
x= lu_apply(A, PermVector, b);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page lu_f lu_f(A)


return LU-Form of matrix A. With U as the upper triangular matrix and L as the lower triangular matrix.

Details: mtl::matrix::lu_f

\code
B= lu_f(A);
L= strict_lower(B);  U= upper(B);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page lu_solve lu_solve(A, b)

Solve \f$ Ax = b \f$ by LU factorization with column pivoting; %vector x is returned

Details: mtl::matrix::lu_solve

\code
x= lu_solve(A, b);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page lu_solve_straight lu_solve_straight(A, b)

Solve \f$ Ax = b \f$ by LU factorization without pivoting; %vector x is returned.

Details: mtl::matrix::lu_solve_straight

\code
x= lu_solve_straight(A, b);
\endcode

For example:

\include lu_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page max_abs_pos max_abs_pos(A)

returns pair(row,col) of maximal absolut entry of %matrix A.

Details: mtl::matrix::max_abs_pos


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/




//-----------------------------------------------------------
/*! \page max_pos max_pos(A)

returns pair(row,col) of maximal entry of %matrix A.

Details: mtl::matrix::max_pos


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page nnz nnz()

returns the number of non zeros of %matrix A.

For example:
\code
int nnz=  A.nnz();
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page num_cols num_cols(A)

returns number of columns of %matrix A.

Details: mtl::traits::num_cols

For example:
\code
int col=  num_cols(A);
int col2= A.num_cols();
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page num_rows num_rows(A)

returns number of rows of %matrix A.

Details: mtl::traits::num_rows

For example:
\code
int row=  num_rows(A);
int row2= A.num_rows();
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page one_norm one_norm(A)

returns one-Norm of %matrix A.

Details: mtl::matrix::one_norm

For example:

\include matrix_norms.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page op_matrix_equal OP=

returns A= B. The dimensions are checked at compile time.
is only correct when A and B have the same number of rows and columns.

\code
A= B;
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page op_matrix_add_equal OP+=

returns \f$ A= A + B \f$. The dimensions are checked at compile time. (\f$ A+= B \f$)

Details: mtl::matrix::add

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page op_matrix_add OP+

returns \f$ A= B + C \f$. The dimensions are checked at compile time.

Details: mtl::matrix::add

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_matrix_min_equal OP-=

returns \f$ A= A - B \f$. The dimensions are checked at compile time. (\f$ A-= B \f$)

Details: mtl::matrix::min

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_matrix_min OP-

returns \f$ A= B - C \f$. The dimensions are checked at compile time.

Details: mtl::matrix::min

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_matrix_mult_equal OP*=

returns \f$ A= A * B \f$. The dimensions are checked at compile time. (\f$ A*= B \f$)

Details: mtl::matrix::mult

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_matrix_mult OP*

returns \f$ A= B * C \f$. The dimensions are checked at compile time. num_cols(B) == num_rows(C).
The dimension of A is num_rows(B) by num_cols(C).

Details: mtl::matrix::mult

For example:

\include matrix_operations.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page qr_algo qr_algo(A,iter)

returns eigenvalues of symmetric %matrix A with qr-algorithm in iter steps.

Deatils: mtl::matrix::qr_algo

For example:

\include eigenvalue_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page qr_sym_imp qr_sym_imp(A)

returns eigenvalues of symmetric %matrix A with symmetric implizit qr-algorithm and Wilkinson shift.

Details: mtl::matrix::qr_sym_imp

For example:

\include eigenvalue_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page rank_one_update rank_one_update(A, v, w)

returns \f$ A= A + v * w \f$.
With %matrix A and %vector v and w of matching dimension.

Details: mtl::matrix::rank_one_update

For example:

\include rank_two_update.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page rank_two_update rank_two_update(A, v, w)

Suppose %matrix A have 2 triangle parts. L is the lower triangle and U the upper triangle part of A.

L -> \f$ L + lower(v*w' + w*v') \f$
U -> \f$ U + upper(v*w' + w*v') \f$

Details: mtl::matrix::rank_two_update

For example:

\include rank_two_update.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page RowInMatrix RowInMatrix<A>

returns row %vector in %matrix A.

Details: mtl::RowInMatrix

\code
RowInMatrix<dense2D<cdouble> >::type v_r2(A[0][iall]);
\endcode

For example:

\include matrix_functions3.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page set_to_zero  set_to_zero(A)

Initializes the entire matrix with zero:
\code
set_to_zero(A);
\endcode
%Matrices are not initialized by default.
Or:
\code
A= 0.0;
\endcode

Details: mtl::matrix::set_to_zero


\include matrix_functions2.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page strict_lower  strict_lower(A)

Returns strict-lower triangle %matrix.

Details: mtl::matrix::strict_lower

\code
strict_lower(A);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page strict_upper strict_upper(A)

Returns strict-upper triangle %matrix.

Details: mtl::matrix::strict_upper

\code
strict_upper(A);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page sub_matrix  sub_matrix(A, row1, row2, col1, col2)

returns submatrix from row1 to row 2 and from col1 to col2 of %matrix A:
\code
sub_matrix(A, 2, 4, 1, 7);
\endcode

Details: mtl::matrix::sub_matrix

Sub-matrices also preserve the const attribute of the referred matrices or sub-matrices:

\include matrix_functions3.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page svd_tol  svd(A, tol)

returns singular-value-decomposition of %matrix A. 3 matrices S, V and D are returned,  with \f$ A= S*V*trans(D) \f$:
\code
boost::tie(S, V, D)= svd(A, 1.e-10)
\endcode
The second argument is optional (default value 1.e-10).

Details: mtl::matrix::svd

\include svd_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page svd  svd(A)

returns singular-value-decomposition of %matrix A. 3 matrices S, V and D are returned,  with \f$ A= S*V*trans(D) \f$:
\code
boost::tie(S, V, D)= svd(A)
\endcode

Details: mtl::matrix::svd

\include svd_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page swap_row  swap_row(A, row1, row2)

returns %matrix A swapped with rows row1 and row2.

Details: mtl::matrix::swap_row

\code
swap_row(A, 2, 4);
\endcode



\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page trace  trace(A)

Returns the trace of a %matrix: 
\code
trace(A);
\endcode

Details: mtl::matrix::trace

\include matrix_functions2.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page trans  trans(A)

The transposed of a %matrix is computed by: 
\code
trans(A);
\endcode
The %matrix A is not altered but a immutable view is returned.


\include matrix_functions2.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page tril  tril(A, i)

Returns lower triangle starting at off-diagonoal i (for compatibility with matlab). 

Details: mtl::matrix::tril

\code
tril(A, i);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page triu  triu(A, i)

Returns upper triangle starting at off-diagonoal i (for compatibility with matlab). 

Details: mtl::matrix::triu

\code
triu(A, i);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page upper  upper(A)

Returns upper triangle %matrix

Details: mtl::matrix::upper

\code
upper(A, i);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/




//---------------------Vector Ooperations


//-----------------------------------------------------------
/*! \page dot_v dot(v,w)

return's scalar-product of %vector v and w.
Dot product with user-specified unrolling defined as hermitian(v) * w.

Details: mtl::vector::dot

For example:

\include dot_test.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page dot_real_v dot_real(v,w)

return's scalar-product of %vector v and w.
Dot product without conjugate with user-specified unrolling defined as trans(v) * w

Details: mtl::vector::dot_real

For example:

\include dot_test.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page infinity_norm_v infinity_norm(v)

returns infinity-norm of %vector v.

Details: mtl::vector::infinity_norm

For example:

\include vector_norm.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page iota_v iota(v)

Iota assigns sequentially increasing values to a %vector v

Details: mtl::vector::iota

For example:
\code
dense_vector<int> v(10);
iota(v);
std::cout<< "v=" << v ;
\endcode
Returns:
v={10C}[0,1,2,3,4,5,6,7,8,9]


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page iota_v1 iota(v, offset)

Iota assigns sequentially increasing values to a %vector v starts with offset-value

Details: mtl::vector::iota

For example:
\code
dense_vector<int> v(10, 5);
iota(v);
std::cout<< "v=" << v ;
\endcode
Returns:
v={10C}[5,6,7,8,9,10,11,12,13,14]


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page max_abs_pos_v max_abs_pos(v)

returns position of maximal absolut entry of %vector v.

Details: mtl::vector::max_abs_pos


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page max_pos_v max_pos(v)

returns position of maximal entry of %vector v.

Details: mtl::vector::max_pos


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page max_v max(v)

returns maximal entry of %vector v.

Details: mtl::vector::max

For example:

\include vector_min_max.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page min_pos_v min_pos(v)

returns position of minimal entry of %vector v.

Details: mtl::vector::min_pos


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page min_v min(v)

returns smallest entry of %vector v.

Details: mtl::vector::min

For example:

\include vector_min_max.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page one_norm_v one_norm(v)

returns one-norm of %vector v.

Details: mtl::vector::one_norm

For example:

\include vector_norm.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_vector_add_equal OP+=

returns \f$ v+= w  (v= v + w)\f$.
Dimensions must agree, otherwise there is a runtime error.

Details: mtl::vector::add

For example:

\include vector_expr.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page op_vector_min_equal OP-=

returns \f$ v-= w  (v= v - w)\f$.
Dimensions must agree, otherwise there is a runtime error.

For example:

\include vector_expr.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_vector_add OP+

returns \f$ v= u + w \f$.
Dimensions must agree, otherwise there is a runtime error.

For example:

\include vector_expr.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page op_vector_min OP-

returns \f$ v= u - w \f$ .
Dimensions must agree, otherwise there is a runtime error.

For example:

\include vector_expr.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page orth_v orth(v)

returns orthogonal and normalized all vectors of %vector v.

Details: mtl::vector::orth

For example:

\include orth_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page orth_vi orth(v, i)

Orthogonalized and normalized vector i from the vector of vectors v.
returns %vector v with orthogonalized i-th entry.

Details: mtl::vector::orth

For example:

\include orth_example.cpp 

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page orthogonalize_factors_v orthogonalize_factors(v)

Opposed to orth the vectors are not normalized. 
An upper matrix with the factors used in the orthogonalization is returned.

Details: mtl::vector::orthogonalize_factors

For example:

\include orth_example.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page product_v product(v)

Returns produkt of all %vector entries.

Details: mtl::product

For example:

\include vector_reduction.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page size_v size(v)

return size of %vector v.

Details: mtl::traits::size

\code
unsigned int len= size(v)
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page sum_v sum(v)

Returns sum of all collection entries (%vector-entries)

Details: mtl::sum

For example:

\include vector_reduction.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page swap_row_v swap_row(v, row1, row2)

Returns %vector v with v[row1]= v[row2] and v[row2]= v[row1].

Details: mtl::vector::swap_row

\code
swap_row(v, 2, 4);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page trans_v trans(v)

return transposed view of %vector v

\code
w= trans(v)
\endcode

Details: mtl::vector::trans

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page two_norm_v two_norm(v)

return two-norm of %vector v.

Details: mtl::vector::two_norm

For example:

\include vector_norm.cpp

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//------Matrix-Vector-Operations-------------------

//-----------------------------------------------------------
/*! \page inverse_lower_trisolve inverse_lower_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$.
The %matrix A must be triangular %matrix otherwise the function can throw an exception.
On matrices with non-unit diagonals, the divisions can be circumvented by inverting the diagonal once with invert_diagonal(A) and then using: 

\code
x= inverse_lower_trisolve(A, b)
\endcode

Details: mtl::matrix::inverse_lower_trisolve


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page inverse_upper_trisolve inverse_upper_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$.
The %matrix A must be triangular %matrix otherwise the function can throw an exception.
On matrices with non-unit diagonals, the divisions can be circumvented by inverting the diagonal once with invert_diagonal(A) and then using: 

\code
x= inverse_upper_trisolve(A, b)
\endcode

Details: mtl::matrix::inverse_upper_trisolve


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page matrix_vector OP*

returns %vector \f$ w= A * v \f$. Same as mult(A, v, w).

Details: mtl::matrix::mult

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page lower_trisolve lower_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$.
The %matrix A must be triangular %matrix otherwise the function can throw an exception.

Details: mtl::matrix::lower_trisolve

\code
x= lower_trisolve(A, b);
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page permutation_av permutation(v)

returns permutation %matrix from corresponding %vector.

See: mtl::matrix::permutation

\code
P= permutation(v);
\endcode

For example:

\include permutation.cpp

For a more detailed explanation see \ref permutation.

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page reorder_av reorder(v)

Returns reorder %matrix from corresponding %vector.

See: mtl::matrix::permutation

\code
P= reorder(v);
\endcode

For example:

\include reorder.cpp

For a more detailed explanation see \ref permutation.

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page upper_trisolve upper_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$ .
The %matrix A must be triangular %matrix otherwise the function can throw an exception.

Details: mtl::matrix::upper_trisolve

\code
x= upper_trisolve(A, b)
\endcode

\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page unit_lower_trisolve unit_lower_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$.
Details: mtl::matrix::unit_lower_trisolve

The %matrix A must be triangular %matrix otherwise the function can throw an exception.
If A has a unit diagonal, the diagonal entries can and must be omitted if the system is solved by:

\code
x= unit_lower_trisolve(A, b)
\endcode


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page unit_upper_trisolve unit_upper_trisolve(A, b)

returns %vector x as solution of \f$ A * x= b \f$.
Details: mtl::matrix::unit_upper_trisolve

The %matrix A must be triangular %matrix otherwise the function can throw an exception.
If A has a unit diagonal, the diagonal entries can and must be omitted if the system is solved by:

\code
x= unit_lower_trisolve(A, b)
\endcode


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/




//-------------Scalar-Vector-Operations---------------------


//-----------------------------------------------------------
/*! \page scalar_vector_mult_equal OP*=

returns %vector \f$ w*= scalar \f$. Same as \f$ w= scalar * w \f$.

Details: mtl::traits::vec_mult_result

\code
w*= 2;
//w= w*2; //throws an exeption.
\endcode

For example:

\include vector_expr.cpp


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-----------------------------------------------------------
/*! \page scalar_vector_div_equal OP/=

returns %vector \f$ w/= scalar \f$. All %vector entries are divided by the scalar.

Details: mtl::traits::div_result

\code
w/= 2;
//w= w/2; //throws an exeption.
\endcode

Details: mtl::vector::div

For example:

\include vector_expr.cpp


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/



//-------------miscellaneous---------------------

//-----------------------------------------------------------
/*! \page iall iall

returns all possible subscripts.

Details: mtl::irange

\code
using mtl::iall;
\endcode

Copy of a col-vector from %matrix with iall:

\code
dense_vector<double>   v(A[iall][0]);
\endcode


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------------------------
/*! \page imax imax

returns the maximum possible index.

Details: mtl::imax

\code
using mtl::imax;
\endcode

\code
dense2D<double>   A(4,4), B(6,6);
A[irange(0,imax)][irange(i,imax)] 
B[irange(0,imax)][irange(i,imax)]
\endcode

Imax has been declared for both matrices. For A imax= 3 and for B is imax= 5.


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/


//-----------------------------------------------------------
/*! \page irange irange(start, end)

returns an index from start to end.

Details: mtl::irange

\code
using mtl::irange;
irange row(2, 4), col(1, 7);
\endcode

Copy of a submatrix with irange:

\code
dense2D<cdouble> B= A[row][col];
\endcode


\if Navigation \endif
  Return to \ref overview_ops &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


*/

//-----------------------------------------

// xxxxxxxxxxxxx


/*! \page faq Frequently Asked Questions


-# \ref faq_not_inserting
-# \ref mtl4_part_of_boost
-# \ref mtl4_in_linux_distributions
-# \ref traits_error


\section faq_not_inserting I always get the error "Assertion failed: !(inserting)" or an exception of type "access_during_insertion" is thrown.

This is the single-most often occurring error people experience when starting with MTL4.

The problem is that certain objects cannot be accessed during insertion, in particular
sparse and distributed matrices.

The following program fragment is wrong:
\code
using namespace mtl;
typedef compressed2D<double> matrix_type;

matrix_type A(5, 5);
matrix::inserter<matrix_type> ins(A);
ins[0][0] << 7.3; // .... more insertions

do_something_with(A);  // TROUBLE!!!
\endcode
In this code, A is used before it is ready.

The insertion is only finished when the inserter is destroyed.
This can be achieved in two ways:
- Defining it in an extra scope; and
- Allocating it dynamically and deleting the pointer (like \ref multiple_insertion "this").

An extra scope is implicitly used when the insertion is performed in a separate function,
as done \ref element_insertion "here".

The easiest way to destroy the inserter is to enclose the insertion in braces:
\code
using namespace mtl;
typedef compressed2D<double> matrix_type;

matrix_type A(5, 5);
{
    matrix::inserter<matrix_type> ins(A);
    ins[0][0] << 7.3; 
}                      // ins is destroyed here
do_something_with(A);  // and A is ready to use
\endcode
For more information read \ref destroy_inserter "this".

\section mtl4_part_of_boost Is MTL4 part of boost?

No. But it has the same directory structure as boost with the intention
of easier inclusion.
Probably, we will apply for a boost revision with the open source edition some day.

\section mtl4_in_linux_distributions Are there plans to become part of Linux distributions

Yes. This is planned for the open source edition.

\section traits_error "xyz" is not defined namespace mtl::matrix::traits

This should not happen and we hope that we already eradicated this error. 
In fact, there were no problems reported recently.

The trouble comes from the fact that there are namespaces traits in mtl as well
as in mtl::matrix.
Within namespace mtl::matrix, the name traits::xyz is searched in mtl::matrix::traits
not in mtl::traits.

The quick solution is to replace traits::xyz by mtl::traits::xyz to nominate the namespace
explicitly.

The best solution is to send us (mtl4@osl.iu.edu) a bug report and we will fix it for everybody.

*/



//-----------------------------------------

// xxxxxxxxxxxxx


/*! \page performance_disclaimer Disclaimer


Unfortunately, the dispatching to BLAS is currently only available for %matrix multiplication.
We work on the extension to other %operations and are not to too proud to accept some generous help.

*/

} // namespace mtl


#endif // MTL_TUTORIAL_INCLUDE