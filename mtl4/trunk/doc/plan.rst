==============================================
 MTL Development Plan (preliminary) and Notes
==============================================

.. sectnum::

.. contents:: Index

Milestone 1
+++++++++++

Create Basic Development Infrastructure
=======================================

Accounts
--------

..

Testing System
--------------

..
      
Documentation System
--------------------

..

Bootstrap Design/Coding
=======================

Iterating between generic interface design and low-level
experiments to characterize performance impact of interface design
decisions.

Develop Fixed Algorithm Size Template Library (FAST)
====================================================

.. 

Develop Basic Linear Algebra Instruction Set (BLAIS) 
====================================================

..


Data Structure Bootstrap
========================

Dense 2D Matrix
---------------

row-/column-major orientations

Dense Vector
------------

..

CSR Matrix
----------

..

Linear Algebra Concept Taxonomy
===============================

..

Algorithm Implementations
=========================


.. role:: concept
   :class: interpreted

Enough support so that vectors model :concept:`VectorSpace` and
vectors + matrices model :concept:`Linear Algebra`.

Expression Templates
====================

Support operator notation for implemented algorithms.

Pre-Release 1
=============

..

Milestone 2
+++++++++++

Eigensystems
============

Givens ROtations
----------------

..

Householder Transform
---------------------

..

Expand Matrix Representations
=============================

Add Storage and corresponding Shape aspects.

Triangular and Banded
---------------------

.. Note:: Triangular can be seen as a special case of banded.

Packed Storage
..............

Applies to banded and triangular shapes

Triangular packed storage
.........................

Applies to triangular shape

BLAS banded storage
...................

Applies to banded shape

Tridiagonal shape
.................

Applies to diagonal orientation


Symmetric
---------

is this really a shape?

.. Note:: re-use triangular packed storage for these

Regular
.......

..

Hermitian
.........

..


Pre-Release 2
=============

..

Milestone 3
+++++++++++

Blocking Dense Matrix Matrix multiply
=====================================

.. Note:: probably involves blocked view of dense matrix

Pre-Release 3
=============

..

Milestone 4
+++++++++++

Sparse Fixed Blocked CSR
========================

New data structure modeling Linear Algebra when combined with
Vector.  Blocking should be exploited for fast Matrix Vector
product

.. Note:: Fast addition may be too hard to do.

Pre-Release 4
=============

..

Milestone 5
+++++++++++

Sparse Variable Blocked CSR
===========================

New data structure modeling Linear Algebra when combined with
Vector.  Blocking should be exploited for fast Matrix Vector
product

.. Note:: Fast addition may be too hard to do.

Pre-Release 5
=============

..


Milestone 6
+++++++++++

Generic LU Factorization
========================

.. Note:: Don't worry about making all combinations fast

SuperLU Implementation
======================

Read the Paper
--------------

Is there special data structure work?

Add :concept:`UnitDiagonal` Matrix Aspect
-----------------------------------------

..

Implement Triangular Solve
--------------------------

..

Pre-Release 6
=============

..


Milestone 7
+++++++++++

Incorporate parallelism in conjunction with parallel BGL


