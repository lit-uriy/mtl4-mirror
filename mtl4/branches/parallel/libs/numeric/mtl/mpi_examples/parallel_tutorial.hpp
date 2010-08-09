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

#ifndef MTL_PARALLEL_TUTORIAL_INCLUDE
#define MTL_PARALLEL_TUTORIAL_INCLUDE

// for references
namespace mtl {

//-----------------------------------------------------------


/*! \page parallel_tutorial Parallel tutorial

-# Warm up
   -# \subpage parallel_hello_world 
   -# \subpage parallel_ostreams
-# Distributed matrices and vectors
   -# \subpage distributed_vector
   -# \subpage distributed_matrix
   -# \subpage distribution_objects
-# Assignment
   -# \subpage distributed_vector_assignment
   -# \subpage distributed_matrix_assignment
-# Operators
   -# \subpage distributed_vector_expr 
   -# \subpage distributed_matrix_expr 
   -# \subpage distributed_matrix_vector_expr
-# Norms (in reductions ??)
   -# \subpage distributed_vector_norms 
-# Reductions
   -# \subpage distributed_vector_reductions 
-# Other Functions
-# Solving Linear Systems
   -# \subpage distributed_krylov_intro
   -# \subpage using_distributed_solvers
-# Parallelization
   -# \subpage agglomeration
   -# \subpage partitioning
   -# \subpage migration
   -# \subpage sparse_alltoall
   -# \subpage boost_mpi_serialization

\todo matrix norms
\todo matrix expressions

*/


//-----------------------------------------




/*! \page parallel_install Parallel installation guide

\section parallel_quickstart Quick start


To compile a parallel MTL4 application you need to
-# Install Boost
-# Install an MPI, e.g., OpenMPI, LAMMPI, or MPICH
-# Build (compile) the Boost serialization and MPI libraries
-# Install MTL4
-# Compile parallel applications with mpiCC
-# Link parallel applications with boost::mpi and serialization libraries
-# Run your application with mpirun
.
This sounds more complicated than it is.


\section install_boost Parallel Installing Boost

is just downloading and unpacking or using a package manager
as in the non-distributed version, see \ref install_boost.


\section parallel_install_mpi Installing MPI

The best way is using a packaging tool under Linux.
If you are using Windows or the MPI version you like is not provided
in your package, go to one of the pages:
- <a href="http://www.open-mpi.org/software/ompi/current">OpenMPI</a>;
- <a href="http://www.mcs.anl.gov/research/projects/mpich2/downloads/index.php?s=downloads">MPICH</a>; or
- <a href="http://www.lam-mpi.org/7.1/download.php">LAM/MPI</a>
.
which are the most important free and portable MPI versions.
Proprietary MPIs are supported if they are standard-compliant.
If you encounter problems contact us.
The binaries of MPI shall contain a wrapper for the C++ compiler,
i.e. mpiCC or mpic++.
This is typically a script that calls a C++ compiler with additional flags.
If this is missing in your MPI installation, you can most likely adapt the
C compiler wrapper mpicc by replacing just the compiler call.


\section parallel_install_boostmpi Building the Boost serialization and MPI libraries

This is described in detail in 
<a href="http://www.boost.org/doc/libs/1_43_0/doc/html/mpi/getting_started.html#mpi.config">the Boost::MPI installation guide</a>.

\section parallel_install_mtl Installing MTL4

It is just downloading 
as in the non-distributed version, see \ref install_mtl.

\section parallel_building Compiling, linking and running with scons or cmake

The easiest way for working with parallel applications
is by using \ref parallel_testing_scons "scons" and soon with
\ref parallel_testing_cmake "cmake" as well. 
If you prefer handling it by hand or your own build system,
 read the following paragraphs
  to understand
a bit of the background.


\section parallel_compiling Compiling parallel applications with mpiCC

MTL4 is extended in a way that one can still compile and run all
non-parallel applications and tests with the Supercomputing Edition.
To this purpose, all source fragments that only compile in the presence
of MPI are wrapped with
\code
#ifdef MTL_HAS_MPI
...
#endif
\endcode
Thus one only need to replace a compile command like the following:\n
<tt>g++ SOURCE_NAME -o EXE_NAME OTHER_FLAGS</tt>\n
by:\n
<tt>mpiCC -DMTL_HAS_MPI <SOURCE_NAME> -o <EXE_NAME> <OTHER_FLAGS></tt>\n
The mpiCC compiler (the underlying C++ compiler) 
must be link-compatible with the one used for 
boost::mpi and boost::serialization.
The best is to use the same mpiCC command as for building boost:mpi.

If you compile in a large data center 
there might be many different MPI and compiler versions in use.
You should ask your administrator for the according
versions.
On the other hand, most centers use <tt>softenv</tt> or <tt>module</tt>
which make it quite easy to select the right version and set up your environment
consistently.
(We find it convenient to write such setup commands in .tcshrc or alike.)

\section parallel_linking Linking with boost::serialization and boost::mpi

The parallel MTL4 uses boost::mpi to pass messages generically.
Boost::mpi in turn uses boost::serialization and both libraries are partially
compiled and need to linked thus to a parallel MTL4 application,
see \ref boost_mpi_serialization.
Note that the library names can vary from platform to platform and
that some distributions of boost generate libraries where the name contains
the compiler (for the before-mentioned link-compatibility) and whether
it is ready for multi-threading.
The two libraries might also be written to different directories.
To facilitate this, some boost distributions create symbolic links with
portable names (e.g. libboostmpi.a); 
something that you could do on your own (if you have write access to the according
directory).
One can use\n
<tt>locate libboost</tt>\n
to find them.
For short, you need compiler flags like:\n
<tt>mpiCC ... -L<SERIALIZATION_DIR> -l<SERIALIZATION_LIB_NAME> -L<BOOST_MPI_DIR> -l<BOOST_LIB_NAME></tt>\n
The non-generic MPI library will be linked transparently by the mpiCC command.

\section parallel_run Running parallel MTL4 applications

You can just run it like any other MPI application with:\n
<tt>mpirun -np <PROCESSES> <OPTIONAL_MPI_FLAGS> <MTL_APP> <FLAGS_OF_YOUR_APP></tt>\n
See the documentation of your MPI for details about execution.
If your boost::mpi or serialization library is dynamically linked,
you will need to <tt>LD_LIBRARY_PATH</tt>, e.g.:\n
<tt>export LD_LIBRARY_PATH=/usr/local/lib/boost</tt>\n
(Again, it is convenient to do this automatically in .bashrc or alike.)


\section Testing

- \subpage parallel_testing_scons.
- \subpage parallel_testing_cmake.

Proceed to the \ref parallel_tutorial.  

*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page parallel_testing_scons Parallel testing with scons

General description of 
testing with scons is provided \ref testing_scons "here".

On this page we describe the extensions regarding the Supercomputing Edition.

\section parallel_scons_compiler Compiler

All parallel tests and examples are compiled with <tt>mpiCC</tt>.
Make sure that <tt>mpiCC</tt> is found in your path.
Parallel tests and examples start by convention with <tt>"mpi_"</tt>.
All other sources are compiled with the regular C++ comiler.

\section parallel_scons_linking Linking



As mentioned \ref parallel_linking "before", the library names of 
boost::mpi and boost::serialization can vary.
Therefore, the build system can be guided by environment variables.
For instance,\n
<tt>export BOOST_SERIALIZATION_LIB=boost_serialization-gcc43-mt-1_37.a</tt>\n
There are three environment variables one can set:
- <b>BOOST_MPI_LIB</b>: the name of the boost::mpi library, by default <tt>boost_mpi.a</tt>
- <b>BOOST_SERIALIZATION_LIB</b>: the name of the boost::serialization library, by default <tt>boost_serialization.a</tt>
- <b>BOOST_LIBPATH</b>: the name of the directory with the two libraries, by default <tt>/usr/local/lib</tt>
.

\section parallel_scons_testing Testing

As mentioned before, the parallel tests in libs/numeric/mtl/test start with
'mpi_'.
Then it follows the number of processes the test is written for, e.g.
<tt>mpi_2_dot_test.cpp</tt>.
This test shall be called with:\n
<tt>mpirun -np 2 mpi_2_dot_test</tt>\n
Of course, parallel MTL4 application can be written in a way that they
work for any number of processes.
It is just easier to check an expected behavior (certain distribution
or given local values on a certain process) if the number of 
processes is known.
One can also use the 'check' flag from our SConstruct, i.e.
build with:\n
<tt>scons -D check=1 mpi_2_dot_test</tt>\n
then the program is compiled, linked and finally started with
<tt>mpirun</tt> on 2 processes.
On a Unix machine you can build and run all MPI tests with\n
<tt>scons -D check=1 `ls mpi*.cpp | grep -v mpi_2_test_mpi_log_test.cpp | sed s/\.cpp//g`</tt>\n
To build and run all test, you type:\n
<tt>scons -D check=1 .</tt>\n
as in the non-parallel MTL4.

\section parallel_scons_dynamic Dynamic linking

If the boost::mpi and serialization libraries are shared libraries,
you need to add their location to <tt>LD_LIBRARY_PATH</tt>, for instance:\n
<tt>export LD_LIBRARY_PATH=/usr/local/lib</tt>\n
You should put this line into an initialization file as <tt>.bashrc</tt>
or <tt>.login</tt>.
This is not only to avoid retyping it but it is important because
<tt>mpirun</tt>  starts your program in another shell potiantially on another 
node of your
cluster or parallel computer and environment variables defined in
the shell before starting <tt>mpirun</tt> are not necessarily available.

\section parallel_scons_additional Additional parallel libraries

MTL4 has an interface to <tt>ParMetis</tt>.
To use <tt>ParMetis</tt>, you must define the environment variable
'PARMETIS_DIR' with its location.
You can additionally define 'PARMETIS_LIB'/'METIS_LIB' for the library name
if it is different than 'libparmetis.a'/'libmetis.a'.



Final remark: do not get used too much to it because future development
will focus mainly on \ref parallel_testing_cmake "cmake".

*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page parallel_testing_cmake Parallel testing with cmake

Coming soon.

*/
//-----------------------------------------------------------


//-----------------------------------------------------------
/*! \page parallel_hello_world Parallel MTL hello world


The following hello world program is an MTL-ish variation of Boost MPI's
hello world:

\includelineno mpi_3_boost_mpi_example.cpp

The program is almost self-explanatory.
- <b>Line 4 and 5</b>: Here we included MTL4 and Boost MPI entirely without affecting compile time significantly.
- <b>Line 9 and 10</b>: Define the MPI environment and a communicator for processes.
  This is part of every application with Boost MPI and also needed here, see also
  <a href="http://www.boost.org/doc/libs/1_43_0/doc/html/mpi/tutorial.html">Boost MPI tutorial</a>.
  The most important is that the environment defined before the first communication happens
  and lives until the last communication is finished.
  The best is to define it as first object in the main function.
- <b>Line 12</b>: A regular non-distributed type like \c dense_vector is replicated on all processes
  and it is not guaranteed that all processes have the same value.
- <b>Lines 14-16</b>: are only executed by process 0. After line 14, %vector \c v is set on rank 0 but
  has still undefined values on the other processes.
- <b>Line 16</b>: The communicator's \c send method allows for sending arbitrary serializable objects.
  Actually, type %traits determine whether the object is really serialized or sent directly, more details 
  \ref boost_mpi_serialization "here".
- <b>Line 20</b>: The counter-part to \c send is \c recv. In the example, all processes receive from 
  process 0, except itself.
- <b>Line 21</b>: The %vector is printed after reception. As the Boost MPI tutorial also states, the 
  order of output is not determined. Sometimes, the output from different processes get even mixed.
  MTL4 provides special output streams for more convenient printing in MPI applications, see \ref parallel_ostreams.

This was a quite simple example of  point-to-point communication.

With a collective communication, the example is even simpler:

\include mpi_3_broadcast_example.cpp

We choose these examples to give a feeling of the generic interplay between Boost MPI and parallel MTL4.

From our own experience, for instance with the linear solvers, most of the parallelism happens in
distributed data.
Operations on distributed matrices and vectors perform the communication transparently for the user.
Some parallel linear solvers have replicated matrices (e.g. the orthogonalization factors in GMRES)
but such matrices are not computed on one process and sent to the others but calculated simultaneously
on each process.

\remark
In the tutorial and the tests we use the convention to name programs according the pattern
"mpi_nn_xyz.cpp" where "nn" is a possible number of processes.
That is one can start the program with "mpirun -np nn mpi_nn_xyz".
Most examples work with arbitrary process numbers while others require a certain
process number for instance when explicitly defined distributions are used.


\if Navigation \endif
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref parallel_ostreams 

*/

//-----------------------------------------


//-----------------------------------------------------------
/*! \page parallel_ostreams Convenient Output in Parallel Programs

In the previous hello world example we have seen that output in parallel programs
is somewhat cumbersome because we have to play with process IDs to avoid printing out 
the same message over and over again (with many processors).
MTL4 has for this purpose some  convenience features.

\section single_output Single Output Stream

The class \ref par::single_ostream serves for printing messages once without 
filtering for a certain MPI rank.
For convenience, the output stream par::sout is defined (as static global object of single_ostream).
One application for this is to print iteration numbers as in the following
example:

\include mpi_3_single_ostream.cpp

If we would write to std::cout the message would appear \p p times on \p p processes.
Here the output looks normal:

\include mpi_3_single_ostream.output

as if no parallelism were present.

An even bigger advantage of par::single_ostream is that it handles distributed data
correctly.
For instance, the distributed vector \p v is printed in the following line:
\code
sout << "Vector v is " << v << '\n';
\endcode
Then the complete distributed vector is written not only the local part of one
process.
For this purpose, distributed data are printed by cooperation of all processes
while non-distributed data like strings are treated by one process.
We will use this capability often in the examples.

In addition to the default constructor, there are three constructors that accept:
- A std::ostream;
- A std::ostream and a boost::mpi::communicator; and
- A std::ostream and a par::base_distribution.
.
Passing an ostream to allows writing conveniently to files, sockets and other streams.

\section multi_output Multiple Output Streams

Another frequent output scheme is writing from all processes indicating the process number.
This can be easily done with the class par::rank_ostream as in the following example:

\include mpi_3_multi_ostream.cpp

This produces the output:

\include mpi_3_multi_ostream.output

Since messages are sent quite fast and terminal output is rather slow it can happen
that the last output line is printed earlier despite the internal synchronization.

Discussion: To improve the quality of output further, this class might be rewritten such that
messages are first sent to one process and completely printed.
Having only one process performing the printing requires more resources and might take more time
so that the more expensive version might be added and not replacing this one.


Remark: The classes currently do not work with std::endl (and other output manipulators). 
Use '\\n' as line end. If you want flushing your output you can use the class' member function,
e.g.:
\code
sout.flush();
\endcode




\if Navigation \endif
  Return to \ref parallel_hello_world &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_vector 

*/

//-----------------------------------------


//-----------------------------------------------------------
/*! \page distributed_vector Distributed Vectors

\section distributed_vector_simple Simple Example

The class vector::distributed is parametrized by a non-distributed vector class
to be extensible for future vector classes (e.g. one with data on GPU memory) 
and to be applicable to third-part vector classes.

The example below shows a simple set up for a distributed vector:

\include mpi_3_distributed_vector.cpp

The vector \c v has a global size of 8.
In this example we do not specify how the vector is distributed but leave this decision to the library.
Note that the <b>local sub-vectors do not overlap</b> as in some other libraries.
Each global entry has a unique location.
(Maybe overlapping will be implemented some day but not in the near future.)

An important difference to non-distributed vectors is the obligation of using 
an inserter to set values.
One can also use an inserter to set non-distributed vectors but this is not mandatory.
Generic implementations that shall be used both on one and on multiple processors must
use an inserter; implementations that always run on a single memory (we have not yet
explicitly programmed for multi-threading and vector element access is not atomic)
can be realized without inserter for the sake of simplicity and clearity.
If you are not familiar with the inserter concept, you should read the
page \ref matrix_insertion.
There is a similar issue between sparse and dense matrices where dense matrices can be set
directly whereas sparse matrices or generic implementations require inserters.
Evidently, vectors use only one index where matrices use two of them but all other
statements on inserters apply in the same manner (e.g. the use of update functors).

In the example, we inserted all values on process 0.
The distributed inserter sends remotely inserted values to the according process.
If one omits the \c if, the insertion would be performed on all processes and
each entry would be inserted \c p times on \p p processes.
In this case, the entry would be overwritten \c p times and the result would be the same.

The insertion is implemented as follows.
If an entry is  local (i.e. it is inserted on the process where it resides) then it is inserted directly.
Remote entries are agglomerated in a buffer and during the destruction of the distributed inserter
sent to the according process and inserted there.

\section distributed_vector_update Additive Distributed Insertion

In simulation applications, the vector entries are computed usually by summation over partial contributions
(from finite elements or cells).
This summation can be easily performed with an inserter:

\include mpi_3_distributed_vector_update.cpp

The inserter is defined with an \c update_plus functor as second template argument.
In this case, all contributions to a vector element are added.
As one can see in the example, one can insert values in an arbitrary process and they
are sent and accumulated transparently.
Of course, it is important in large-scale applications that the majority of insertions
is local to avoid excessive communication.
In a simulation context that means that a vertex is located in one of the processes 
of the containing elements or cells.
Please note that dense data structures are not automatically initialized and that one
must set the vector to 0 (e.g. in the constructor) before starting an additive insertion.

The distributed vector class has two template arguments: the local vector class (as seen in the
example) and a distribution type.
The latter is by default block distribution.
So far we left it to the library to decide how to distribute the vector.
MTL4 distributes the entries almost uniformly over the processes with potentially one 
more on first processes.
As we have seen above, 8 entries on 3 processes are split into 3 + 3 + 2.
One can also choose any other block distribution as long as it is large enough for the vector.

\section distributed_vector_block Customized Distribution of a Vector

A block distribution is defined by \c p+1 entries for the first global index 
of each process and the maximal global size.

\include mpi_3_users_block.cpp

In this example, the vector of size 8 is distributed as 2 + 5 + 1.
If we had a vector with global size 7 or less, the sub-vector on process 3 would be empty.
If the global size would be 9 or higher, the construction of the distributed vector would fail.
More information on distribution is found in section \ref distribution_objects.

The output of the program is:

\include mpi_3_users_block.output

\remark
Please not that this example does not run with more or less than 3 processes.
If one takes one or two processes, the distribution does not allow to
build a vector of size 8.
When more than 3 processes are taken, the distribution is not defined
by an array of size 4.
Likewise, other examples with an explicit block distribution does not work 
with arbitrary process numbers.

\section distributed_vector_temporaries Temporary Vectors


If one needs a temporary vector in an algorithm it is not always correct
to create one of the same size.
The following code demonstrates the right and wrong way to construct temporaries.

\include mpi_3_vector_temporary.cpp

We do not want to bother the reader with the ugly output of uninitialized vectors.

There are two vectors serving as reference objects:
- u: a vector of size 8 distributed by default as 3 + 3 + 2 and initialized to the value 4.0;
- v: a vector of size 10 with user distribution 2 + 5 + 3 and initialized to the value 5.0;

The four temporary variables are defined as follows:
- w: has the same size as u. Since no distribution is defined, the default distribution is
  used like for u. Therefore, they are compatible. 
- x: has the same global size as v. Since no distribution is defined the default distribution is
  used which is 4 + 3 + 3. Thus, x and v are not compatible.
- y: is a copy of v. It copies global size, distribution and values. It is evidently compatible
  to v but copying its values is a surplus effort in many algorithms.
- z: has the same size and distribution like v. In contrast to y, its entries are still not
  set.

\remark
Vectors with the same global size and  default block distribution are always compatible.
This applies also for the other distribution types in MTL4.
Of course, we cannot give such guarantees for user-defined distribution types if
they were undeterministic.
Vectors with different distribution types are always considered incompatible
although there are rare cases where they can be compatible.
The copy constructor always yields vectors compatible to their sources but with extra
copy effort.
Constructing new vector with vector::resource is the generic manner to create temporaries without
copying them.
In the sequential MTL4, vector::resource was therefore introduced retroactively to provide
a unified interface.
Future MTL4 editions, e.g. for accelerators, will also support vector::resource to
establish temporaries with compatible resources, e.g. GPU memory.
For short, rely on vector::resource to provide sufficient information on an object to
create a new object that is compatible to its reference object.
A corresponding resource function will also be added for matrices to create temporaries
generically.

\if Navigation \endif
  Return to \ref parallel_ostreams &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_matrix 

*/


//-----------------------------------------


//-----------------------------------------------------------
/*! \page distributed_matrix Distributed Matrices

\section distributed_matrix_simple Simple Example

If one do not want to customize every parameter, distributed matrices can be handled much like local ones, see example:

\include mpi_3_distributed_matrix.cpp

MTL4 will distribute the matrix for you and you can see the distribution in the output:

\include mpi_3_distributed_matrix.output

Matrices are distriubted row-wise, i.e. each row is owned entirely by one process.
In our example we have the default distribution 3 + 3 + 2.

The print-out of distributed matrices also shows local sub-matrices:
On process 0, there are two 3 by 3 matrices and one 3 by 2 matrix.
This sub-matrices have the type that is indicated by the first template argument of matrix::distributed.

\remark
In later versions, we will probably distinguish between diagonal and off-diagonal block matrices.
Diagonal blocks will still have the type of the template argument while off-diagonal blocks might
have a different type (determined from the template argument by a meta-function).
The reason of this type distinction is that off-diagonal blocks are usually clearly sparser 
and we are developing matrix types to cope with such sparsities.

In distributed matrices,  some or most of the matrix blocks can be empty.
Especially large sparse matrices on many processes originating from a domain decomposition have usually a small number of non-empty sub-matrix
blocks and many empty ones.
Such empty matrix blocks are not stored.
When a distributed matrix is printed, they cannot be omitted as this would confuse the presentation of columns.
Instead they are 
printed as stars for distinction from non-empty blocks like
 in the example.
There, only the diagonal blocks exist and all off-diagonal blocks are empty (not surprising for a diagonal matrix).

\section distributed_matrix_insertion Insertion

Inserting values into a distributed matrix is not different from the insertion into a non-distributed.
However, one should pay attention to avoid redundant insertion.
A program that performs the same insertion operations on each process will
insert every entry \p p times when the program runs with \p p processes.
With an additive insertion this would result in having the values multiplied by \p p and
in an overwriting insertion the matrix would be the same but a significant overhead
would be created.

In the following example, only two of the three processes insert values:

\include mpi_3_distributed_matrix_insertion.cpp

The resulting distributed matrix is:

\include mpi_3_distributed_matrix_insertion.output

In parallel MTL4 it does not matter where entries are inserted.
However, for performance reasons, one should try to insert most values locally.
In parallel simulation programs this can be achieved by distributing the unkowns (entries of matrices and vectors)
similar to the cells or elements so that most results of the assembling are inserted locally.
Only on grid points shared 
between sub-domains some communication is necessary (as mentioned before we do not support overlapping yet).

Of course, the inserter of distributed matrices can be parametrized with an update functor as any other inserter, e.g.:
\code
  typedef mtl::matrix::distributed<mtl::compressed2D<float> >  matrix_type;
  matrix_type A(7, 7);
  mtl::matrix::inserter<matrix_type, mtl::operations::update_plus<float> > ins(A);
\endcode



\section distributed_matrix_custom User-Defined Distribution of Matrices

We already mentioned that our matrices are distributed row-wise.
This distribution can be defined by the user.
The user can also define a column distribution.
This affects how the matrix is constructed internally from sub-matrix blocks.
It is also important to implement operations when not all  data
structures are distributed equally, see \ref distributed_mvp.

\subsection distributed_matrix_custom_homo Homogeneous Distribution

 In the following example, we distribute the 7 rows as 4 + 2 + 1.
This is done as for entering the partial sums into a std::vector and passing this to
a block distribution object:

\include mpi_3_matrix_homogeneous_insertion.cpp

The resulting matrix is:

\include mpi_3_matrix_homogeneous_insertion.output

Note that the columns are split in the same manner into the blocks of sub-matrices.

\subsection distributed_matrix_custom_hetero Heterogeneous Distribution

The column-wise splitting does not need to be identical with the row distribution
as in the next example:

\include mpi_3_matrix_heterogeneous_insertion.cpp

The column are split as 5 + 2 + 0:

\include mpi_3_matrix_heterogeneous_insertion.output

The fact that the last blocks of each process contain no column does not cause
errors in MTL4.
Likewise, empty ranges in the row distribution - such that certain processes have
no part of the distributed matrix - does not cause problems in parallel programs.
However, empty ranges in distributions most likely lead to sub-optimal parallel performance.

\subsection distributed_matrix_custom_flexible Arbitrary Distribution

In our examples and tests we focus on block distribution because this is the
distribution most frequently used in scientific applications.
Other distributions - including user-defined - can be chosen for full flexibility.

In the following example, we use a block-cyclic row distribution while the columns
are spread cyclically into sub-matrices:

\include mpi_3_flexible_insertion.cpp

\remark
Printing such matrices is currently not correct. 
Sub-matrix blocks are treated as if their  global row and column indices would be contiguous.
Thus, only matrices and vectors with block distributions are printed correctly
at the moment.

\todo
The example crashes now with index out of range. Will be fixed quickly.



\if Navigation \endif
  Return to \ref distributed_vector &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_vector_assignment 

*/


//-----------------------------------------


/*! \page distributed_vector_assignment Distributed %Vector Assignment


Vectors assignments are in most cases performed by expression templated 
(see \ref vector_expr) like in the non-distributed case, as simply as:
\code
  v= 3 * w + u;
\endcode

The only difference is that not only the size must be compatible
but also the distribution, see \ref distributed_vector_temporaries.

Functions that return vectors are subject to move semantics (see \ref copying
and \ref move_semantics).



\if Navigation \endif
  Return to \ref distributed_matrix &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_matrix_assignment 

*/


//-----------------------------------------


/*! \page distributed_matrix_assignment Distributed %Matrix Assignment

The following example illustrates the simplicity of distributed
%matrix assignment:

\include mpi_3_matrix_assign.cpp

\todo Insert the entries only once for A= alpha when A is distributed

The matrix A has in this example a user-defined distribution while matrix B
is default-constructed.
The default constructor creates a matrix with global size 0 by 0.
Such matrices are compatible to all matrices no matter what size and distribution,
see \ref stem_cells.
After the assignment, B has the same size and distribution as A.

If we would assign C then, it would cause an error because the
distributions are not compatible.

The distributed matrix class uses \ref move_semantics like the sequential ones
and of course the matrices are not shallowly copied, cf. \ref shallow_copy_problems.


\if Navigation \endif
  Return to \ref distributed_vector_assignment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_vector_expr 

*/


//-----------------------------------------------------------
/*! \page distributed_vector_expr Parallel %Vector Expressions

Distributed vectors can be used in expressions in the same fashion as non-distributed ones:

\include mpi_3_vector_expr.cpp

One can see that sequential programs consisting of vector operations work directly with distributed
vectors without modifications.

Here is a general description for the interested reader how the expressions are evaluated:
- The parallel computation is performed without communication.
- At the beginning the distribution of the operands and the target vector are compared. If the
  the global size or the distribution is different
  then an exception is thrown (distributions of different types can even
  cause compile errors).
  The test is only performed in debug mode.
- If the target vector has global size 0 -- e.g. because it is default constructed -- then the global
  size and the distribution of the source is copied.
  Different distribution types cause an error.
  In the example, the distribution of w is set in the first assignment while the second
  assignment of w verifies that the distributions are equal.
- After the check or initialization, it is clear that all (element-wise treated) 
  vectors in an expression have 
  the same distribution and the operation is performed on local sub-vectors.
- Only operands that are evaluated together need the same distribution and global size.
  For instance, in the operation
  \code
  x= dot(v, w) * u;
  \endcode
  the dot product is computed first and afterward used as scalar operand.
  Therefore,  the vectors v and w must be compatible and u must be compatible with x.
  The vectors x and v can be distributed differently or have a different global size.

Expressions on mixed types are allowed as the last statement shows.
After multiplying the real vector x with the complex constant i the resulting
object is a complex vector (actually a view that behaves like a vector).

\remark
In the example, a complex double is multiplied with a float.
Note that this does not work with standard C++ but is an extension of MTL4.

\todo Talk of mixed complex operators in trunk tutorial.

\if Navigation \endif
  Return to \ref distributed_matrix_assignment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_matrix_expr 


*/



//-----------------------------------------------------------
/*! \page distributed_matrix_expr Parallel %Matrix Expressions

Under development.


\if Navigation \endif
  Return to \ref distributed_vector_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_matrix_vector_expr 


*/


//-----------------------------------------------------------
/*! \page distributed_matrix_vector_expr Parallel %Matrix %Vector Expressions

The following example program illustrates that the parallel matrix vector product can
be easily programmed with the multiplication opeator:

\include mpi_3_matrix_vector_product.cpp

In the code above the vector u had no global size and distribution.
In this case, those parameters are set with the global row number and row distribution of the 
matrix.



\if Navigation \endif
  Return to \ref distributed_matrix_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_vector_norms 


*/


//-----------------------------------------------------------
/*! \page distributed_vector_norms Parallel %Vector Norms




\if Navigation \endif
  Return to \ref distributed_matrix_vector_expr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_vector_reductions 


*/


//-----------------------------------------------------------
/*! \page distributed_vector_reductions Parallel %Vector Reductions




\if Navigation \endif
  Return to \ref distributed_vector_norms &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distributed_krylov_intro 


*/


//-----------------------------------------------------------
/*! \page distributed_krylov_intro Parallel Krylov Methods




\if Navigation \endif
  Return to \ref distributed_vector_reductions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref using_distributed_solvers 


*/


//-----------------------------------------------------------
/*! \page using_distributed_solvers Using Parallel Solvers




\if Navigation \endif
  Return to \ref distributed_krylov_intro &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref distribution_objects 


*/

//-----------------------------------------------------------
/*! \page distribution_objects Distribution Objects



\if Navigation \endif
  Return to \ref using_distributed_solvers &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref partitioning 

*/



//-----------------------------------------------------------
/*! \page agglomeration Agglomeration

This page is more about anti-parallelization but this is needed sometimes as well,
especially for debugging.

Assume you have distributed matrices and vectors and performed a freshly implemented 
parallel computation on them.
The first thing that comes to mind for testing the implementation is to run the same
computation with a reliable sequential implementation (unless the parallel version
is purposefully implemented differently to treat savings in communication for 
weakened numeric behavior).

For this purpose the data are agglomerated first on one process like the distributed matrix
in the following example:

\include mpi_3_matrix_agglomerate.cpp

In the example, we determined the local type from the distributed one. 
One might argue that the opposite is easier, which is true.
However, the meta-function is needed when the distributed type is a template argument:
\code
template <typename D>
void do_something_agglomeratively(const D& x)
{
   what is the local type of D???
}
\endcode

The function matrix::agglomerate can be equipped with a second argument to
agglomerate on a another process.

Accordingly, vectors are agglomerated with vector::agglomerate.
Of course, both functions are found with argument-dependent lookup (ADL).

Testing wether a unary function behaves equally (besides rounding errors)
can be implemented as follow:
\code
distributed_argument_type dx= ...;
distributed_result_type   dy= f_par(dx);

local_argument_type lx= agglomerate(dx);
local_result_type   ly= f_seq(lx), 
                    dy_agg= agglomerate(dy);

assert(some_norm(ly - dy_agg) < eps);
\endcode

In category theory, one would just say that f and agglomerate commute up to rounding errors.


\if Navigation \endif
  Return to \ref distribution_objects &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref migration 


*/

//-----------------------------------------------------------
/*! \page partitioning Partitioning




\if Navigation \endif
  Return to \ref distribution_objects &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref migration 


*/

//-----------------------------------------------------------
/*! \page migration Migrating Matrices and Vectors




\if Navigation \endif
  Return to \ref partitioning &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref sparse_alltoall 


*/


//-----------------------------------------------------------
/*! \page sparse_alltoall Sparse All-to-all Communication




\if Navigation \endif
  Return to \ref migration &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref boost_mpi_serialization 


*/





//-----------------------------------------------------------
/*! \page boost_mpi_serialization Boost MPI and Serialization

Generic message passing means in this context that distributed matrices and
vectors can consist of every data type that is serializable,
see <a href="http://www.boost.org/doc/libs/1_43_0/libs/serialization/doc/index.html">
boost serialization tutorial</a>.
Intrinsic data types and MTL4 containers of intrinsic data types 
are serializable.

Moreover, Boost MPI allows sending intrinsic types and arrays thereof directly
without serialization, when the meta-function mpi::boost::is_mpi_datatype evaluate to true 
(see <a href="http://www.boost.org/doc/libs/1_43_0/doc/html/mpi/tutorial.html#mpi.performance_optimizations">here</a>).
We optimized PMTL4 such that  matrices and vectors of such MPI data types are also
communicated without serialization.


According to our experience, 
most scientific applications run only on parallel computers or clusters with compatible processors (regarding the
bit-wise representation of data).
To avoid surplus MPI_Pack/MPI_Unpack compile your application (and your Boost MPI)
with the macro <tt>BOOST_MPI_HOMOGENEOUS</tt>.

\todo serialization of user types

\todo user types as mpi datatype.

\if Navigation \endif
  Return to \ref sparse_alltoall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Proceed to \ref parallel_overview_ops 

*/


//-----------------------------------------


//-----------------------------------------------------------
/*! \page parallel_overview_ops Overview of parallel operations


Todo: check which work in parallel

-# %Matrix Operations
   - A[range1][range2], not implemented because too expensive
   - \subpage adjoint
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
   - \subpage max_v
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
  Return to \ref boost_mpi_serialization &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \ref parallel_tutorial "Table of Content" &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 

*/

//-----------------------------------------------------------

//-----------------------------------------------------------
/*! \page par_mat_vec_expr  A[range1][range2]


This is operation is very limited with distributed matrices because
an implementation with the same functionality as in the non-distributed
case would be unbearably expensive.

*/


} // namespace mtl


#endif // MTL_PARALLEL_TUTORIAL_INCLUDE
