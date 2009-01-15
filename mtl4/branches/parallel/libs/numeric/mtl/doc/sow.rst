=========================================================
 Statement of Work for MTL Project with Boost Consulting
=========================================================

.. raw:: html

  <style type="text/css">dt { font-style: italic }</style>

.. sectnum::

.. contents:: Index

The project is a redesign and reimplementation of the MTL library
first developed by Andrew Lumsdaine, Jeremy Siek, and Lie-Quan Lee.

Implementation guidelines
=========================

Code should be standards-conformant C++ with minimum of
workarounds (assume in the future compilers will become more
conformant). 

Re-use SL and Boost -- but balance against run-time performance
and compile times.  Performance is very important, compile
times only mildly important.

In general, elegance and adherence to generic programming
principles is most important (more important than performance or
compile time).

Use of SL and Boost should only cost minimal (preferably no)
performance loss, and not add significant compile time.

The design/implementation may include development vs. release
modes.  For development mode, fast compile time is most
important and execution time is secondary.  For release mode,
execution time is most important and compile time is
secondary.  End user should be able to switch between these
with a preprocessor flag (e.g.), not change their code.

General Requirements
====================

- Code shall be thread-safe and be able to accommodate shared and
  distributed memory concurrency.  OSL will provide ongoing
  guidance for this.

- There shall be no memory leaks. 

- All code shall compile with maximum pickiness for each supported
  compiler and few or no warnings. Warnings that are not easy to
  eliminate shall be documented.

Supported compilers
--------------------

- **GNU C++ 3.3** and later
- **Visual C++ 7.1** and later
- **Intel C++ 8** for linux
- **Intel C++ 8** for Windows (VS 7.1 and later emulation)

- **IBM xlc++ 6.0** to be used only for limited performance
  assessments on G5 systems.  If language compliance is
  sufficiently advanced in 6.0 (or later versions), use to compare
  performance of kernels, basic linear algebra algorithms, and
  expression templates (in that order).

- **KAI C++ 4.0** does not need to be fully supported, but may be
  used for abstraction penalty testing on x86 linux systems.
  Support only needs to be sufficient for abstraction penalty
  testing, not complete MTL support.  If porting effort becomes
  significant, only a subset of the abstraction penalty tests is
  required.

Supported Operating Systems
---------------------------

* Linux RH 8 and above (or equivalent)
* Mac OS X
* Windows XP cygwin
* Windows XP native

Supported Architectures
-----------------------

* ia32
* PowerPC G5

Deliverables
============

.. role:: concept
   :class: interpreted

Concept Taxonomies
------------------

Linear Algebra.
  A description of requirements on C++ implementations of
  mathematical concepts such as
  :concept:`VectorSpace` and :concept:`LinearOperator`.

Implementation.
  A set of refinements of linear algebra concepts that expose
  compile-time knowledge that can be exploited for optimization
  purposes.  These concepts capture information such as

  - Shape
  - Orientation
  - Symmetry
  - Sparsity
  - Blocking
  - Upper / Lower storage
  - Unit diagonal
  - Efficiency of insert operations

Low-level Kernels
-----------------

Fixed-size generic algorithms (as in FAST) and linear algebra (as
in BLAIS) that can provide high performance within the `Generic
Algorithms`_.  These must be available to users.  Consider
eliminating duplication with variable-size algorithms by having
single algorithm implementations that can use fixed-size
optimizations when the size happens to be encoded in a type.

Generic Algorithms
------------------

These algorithms may be specialized for certain types or concept
refinements, but they have a generic implementation, as opposed
to being "intrinsic" to the types (e.g. begin()).

Vector Operations
.................

Data Movement.
  - set to scalar
  - copy A to B
  - swap
  - scatter, gather

Reduction.
  These operations produce a scalar (or iterator to a scalar)

  - dot, sum, sum-of-squares
  - 1-norm, 2-norm, inf-norm
  - find-max, find-min

Arithmetic.
  * vector-vector elementwise operations:  ``+``, ``-``, ``*``,
    ``/``, ``+=``, ``-=``, ``*=``, ``/=``

  * vector-scalar operations: ``*=``, ``/=``, ``*``, ``/``

Matrix Operations
.................

Data Movement.
  - set to scalar
  - copy A to B
  - swap
  - transpose

Norms.
  - 1-norm
  - Frobenius
  - inf-norm

Arithmetic.
  * matrix-matrix elementwise operations:  ``+``, ``-``, ``*``,
    ``/``, ``+=``, ``-=``, ``*=``, ``/=``

  * matrix-scalar operations: ``*=``, ``/=``, ``*``, ``/``

  * matrix-vector operations: ``*=``, ``*``,
    multiply-accumulate (inplace and functional)

  * matrix-matrix operations: ``*=``, ``*``,
    inplace multiply-accumulate,
    inplace diagonal multiply,
    inplace diagonal multiply-accumulate.

Rank Updates.
   rank 1 update, rank 1 conj, rank 2 update, rank 2 conj

Triangular Solves.
   - matrix-vector inplace
   - left matrix-matrix inplace
   - right matrix-matrix inplace

Trace.
   Sum of diagonal elements

Eigensystems.
  - Givens Rotation: generate, apply
  - Householder Transform: generate, apply left, apply right

Data Structures
---------------

Vectors
.......

Dense and sparse.

Matrices
........

All matrices should be available in row-major and column-major
variants.

Dense.
  * Regular 
  * Blocked
  * Packed (as defined in BLAS/LAPACK docs)
  * Diagonal and block diagonal
  * Upper and Lower Triangular
  * (BLAS) Banded
  * Tridiagonal
  * Symmetric (regular/Hermitian)
  * Unit Diagonal
  * Hierarchical (Morton ordered, quadtree) [#morton]_

Sparse. [#nist_sparse]_
  * Compressed sparse row/column (CSR/CSC)
  * Block CSR/CSC
  * Variable block CSR / CSC 
  * Superlu and/or dist  (symmetric precursor to superLU) format
  * BGL adjacency graph adapters.  List- and map-based sparse
    matrices must be adapted to mutable adjacency graphs.

.. [#nist_sparse] Compressed sparse formats are defined in NIST
   sparse BLAS: http://math.nist.gov/spblas/

.. [#morton] Work with OSL and David Wise researchers to determine
   specifics, including required data structures and applicability
   of algorithms and concepts.

Expression Templates
--------------------

Provide mathematical notation interface to algorithms.  Operators
+, - (binary and unary), ``*`` (and ``/`` for scalars), as well as
``+=``, ``-=``, ``*=``, ``/=`` are required.  Integrating other
(functional) operations into the expression template engine is
optional.

Test Suite
----------

- Comprehensive to test reasonable combinations of data types and
  algorithms for compilation and execution correctness. 

- Automatic test result dashboard script for automagically
  generating test result dashboard (ala Boost or VTK) [#dash]_

.. [#dash] This was marked optional in the last version of this
   document, but it's important enough to the overall success of
   the project that I've made it a requirement.

Performance Tuning Framework
----------------------------

Similar in spirit to the one used by ATLAS, this framework computes
optimal blocking sizes and number of blocking levels through
empirical tests.  Effectiveness must be demonstrated for dense
matrix-matrix product computation on x86 and PPC G5 processors.

LAPACK Interface
----------------

Basic functions, similar to existing MTL 2 allowing access to
LAPACK functions with MTL data types and element types ``float``,
``double``, ``complex<float>``, ``complex<double>``.  E.g., the MTL
algorithm::

  template <class LapackMatA, class LapackMatB, class VectorInt>
  int gesv (LapackMatA& a, VectorInt& ipivot, LapackMatB& b)

will dispatch to the appropriate Fortran ``gesv`` function for the
matrices' element types.

The following lapack algorithms must be supported:

a) ``gecon``
b) ``geev``
c) ``geqpf``, ``geqrf``
d) ``gesv``
e) ``getrf``, ``getrs``
f) ``geequ``
g) ``orglq``, ``orgqr``

Documentation
-------------

Output format in HTML and PDF; source format TBD, at discretion of
Boost Consulting.

Project Plan and Status.
  This document shall be maintained throughout the life of the
  project.  It is expected to evolve as more of the project is
  completed.  See preliminary project plan attached `here`__.

__ plan.html

User documentation.
  For end users of the package (e.g., computational scientists).
  Should indicate to the user what kinds of combinations of types
  under which operations are efficient (or not).

Developer documentation. 
  For programmers who would want to extend MTL.

Comments.
  Implementation details are explained in-code.

Test Report.
  Describes each test (briefly) and shows pass/fail results of
  each  for supported compilers, OS, and architectures.

Performance Tuning Report.
  Describes use of tuning framework and shows results of its
  demonstration. 

Configuration and Installation
------------------------------

Portable configuration and installation scheme; details TBD in
cooperation with OSL.  By default we will use GNU autotools on UNIX
and InstallShield on Windows.  Release packages to be produced in
cooperation with OSL.

Optional Items
==============

These items are considered low priority, and may be included or
eliminated at the discretion of OSL at appropriate places in the
process.  Boost Consulting will notify OSL when those opportunities
arise.

Blocked Triangular dense matrix.
  Upper and Lower diagonal

Linked list and/or map-based sparse structure.
  These are structures that are more efficient to construct
  and mutate than the compressed sparse formats.  Execution is
  not as efficient however.  Some two-phased notions (convert
  from one format to another depending on task at hand) may be
  appropriate.

Matrix Type Generator.
  Vaguely defined as anything that makes it easier for
  users to come up with matrix types.

LAPACK Algorithms.
  Native C++ versions of LAPACK algorithms, tuned for competitive
  efficiency. 

IA64 Support.
  It's unclear what specific support should be required for IA64
  processors, but testing at least would be needed.




