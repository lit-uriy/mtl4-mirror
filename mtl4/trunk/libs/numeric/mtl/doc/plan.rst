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

Larry Meehan reports that accounts have been set up at:

- magrathea.osl.iu.edu   (main NFS file server for user directories)
- milliways.osl.iu.edu (linux -- hosts boost mailing lists)
- eddie.osl.iu.edu   (currently down)
- vogon.osl.iu.edu  (linux)
- earth.osl.iu.edu  (solaris)
- deep-thought.osl.iu.edu  (solaris)
- frood.osl.iu.edu and some other Mac systems.

(2004-11-9)

Testing System
--------------

We'll be using `Boost.Build version 2`_ (BBv2) for all
building/testing.  I've invested great deal of time recently in
trying to grok BBv2_, and am working closely with Vladimir
Prus (the primary maintainer) to ensure that its documentation is
comprehensible, which means going through a massive review/edit
cycle.

A project has just been started to rewrite Boost.Build in Python_,
hopefully with a Scons_ substrate.  The rewrite should yield many
advantages, not least freeing Boost.Build developers from the
shackles of the odd language built into `Boost.Jam`_, and much
smarter target updating logic.  Scons_ is a wonderful build system,
and several projects hosted at OSL_ have apparently started using
it.  That said, it is very low-level; we want the high-level and
platform-/compiler-neutral functionality of Boost.Build.

.. _BBv2: http://boost-consulting.com/boost/tools/build/v2/
.. _`Boost.Build version 2`: BBv2_
.. _Scons:  http://scons.sourceforge.net/
.. _OSL: http://osl.iu.edu/
.. _Python: http://www.python.org
.. _Boost.Jam: http://boost-consulting.com/boost/tools/build/jam_src/index.html

(2005-1-25)

Documentation System
--------------------

Right now we are using Docutils_ and reStructuredText_ for
documentation.  We have an automated system called litre (“\
**lit**\ erate **re**\ Structuredtext”) for extracting and testing
C++ examples.  Serious consideration is being given to the idea of
moving to quickbook_, not least because we expect the codebase to
be more understandable and maintainable.  Translating litre to
quickbook_ will require generating some Python_ bindings, though,
as some scripting language integration is crucial.

.. _Docutils: http://docutils.sourceforge.net
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _quickbook: http://spirit.sourceforge.net/dl_docs/quickbook_doc/doc/html/index.html

(2005-1-25)

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


