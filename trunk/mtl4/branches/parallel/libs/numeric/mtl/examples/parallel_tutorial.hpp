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
Note that the library names can very from platform to platform and
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



*/
//-----------------------------------------------------------



//-----------------------------------------------------------
/*! 
\page parallel_testing_cmake Parallel testing with cmake

Coming soon.

*/
//-----------------------------------------------------------


//-----------------------------------------------------------

/*! \page parallel_tutorial Parallel tutorial


*/


//-----------------------------------------


//-----------------------------------------------------------
/*! \page boost_mpi_serialization Boost MPI and Serialization

Generic message passing means in this context that distributed matrices and
vectors can consist of every data type that is serializable,
see <a href="http://www.boost.org/doc/libs/1_43_0/libs/serialization/doc/index.html>
boost serialization tutorial</a>.
Intrinsic data types and MTL4 containers of intrinsic data types 


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
