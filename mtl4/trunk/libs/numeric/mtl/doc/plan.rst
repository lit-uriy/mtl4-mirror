==============================================
 MTL Development Plan (preliminary) and Notes
==============================================

.. sectnum::

.. admonition:: Current Status

  My work on FAST_ has taken me into the realm of `how to represent
  sequences`__ with cursors and property maps, and `how to dispatch
  algorithms.`__

__ ../../../../sequence/doc/html/index.html
__ algorithm_dispatch.rst

.. contents:: Index

.. role:: concept
   :class: interpreted


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

.. _FAST:

As part of this work I anticipate having to create an inheritance
hierarchy in MPL_ so that, e.g. ``int_<N>`` is derived from
``integral_c<int, N>``, and add operators that can do simple
arithmetic on these types.

.. _MPL: http://www.boost.org/libs/mpl

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

In which we define concepts such
as :concept:`Ring`, :concept:`Field`, :concept:`LinearOperator`,
:concept:`LinearAlgebra`, :concept:`TransposableLinearOperator`, :concept:`AbelianGroup`,
:concept:`HilbertSpace`, :concept:`BanachSpace`, :concept:`VectorSpace`,
and :concept:`R-Module`.


Dealing with the Imprecision of Floating-Point
----------------------------------------------

(2005-1-27)

Traditional mathematical concepts are defined in terms of
calculations on pure numbers that exhibit no rounding error, but
the number types we use every day in numerical linear
algebra (e.g., ``float`` and ``double``) don't behave quite that
well [High02]_. In Section 7.1, subsection **Equality** of Jeremy Siek's
`preliminary documentation`_ for his early prototype of this
project, the notation

   boost/tools/build/jam_src/

  *a* =\ :sub:`ε` *b*

was used to mean “|\ *a* - *b*\ | < ε where ε is some appropriate
small number for the situation (like machine epsilon).”  The
problem with that is that it's too fuzzy.  In particular, according
to Andrew Lumsdaine, ordinary floating-point numbers don't actually
model :concept:`Field` when notation is used to describe the
concept.

One approach to this issue might be to expel the notion of
imprecision from the concept taxonomy.  Concepts
like :concept:`Field` would be require true equality, and we'd deal
with the imprecision of floating-point by saying, that if an
algorithm requires one of its arguments to model :concept:`Field`
and you pass a ``double`` (which isn't quite a model of
:concept:`Field`), then naturally the algorithm doesn't produce the
promised result.  Instead, if you pass an approximation of a
:concept:`Field` to the algorithm it produces some approximation to
the specified result.

That approach is unsatisfying because the error bounds of any
algorithm when used with real-life floating datatypes can be
calculated, and we'd like our algorithm specifications to be able
to make some promises about the magnitude of those errors.
Naturally, if you have violated an algorithm's requirements by
passing a ``float`` where it expects a pure :concept:`Field`, the
algorithm can't make any promises at all about the result!  Looked
at from the other side, if the algorithm can make some guarantees
about the result it produces for some input, then whatever the
specification says, the input must clearly satisfy some real,
underlying requirement.

Only by keeping floating types in the concept taxonomy can we
sensibly make guarantees about the precision of algorithms
operating on those types.  We assert that ``float`` and ``double``
model a concept called
:concept:`FieldWithError` [#fieldwitherror]_, of which
:concept:`Field` is a refinement that requires perfect precision.
Similar “-:concept:`WithError`\ ” counterparts exist for all the
basic algebraic concepts.  Just
as algorithms like ``std::binary_search`` require
:concept:`Forward Iterators`` but make stronger efficiency
guarantees when passed :concept:`Random Access Iterators``,
numerical algorithms can require their arguments to model the
imprecise “-:concept:`WithError`\ ” concepts and make stronger
precision guarantees when operating on models of precise algebraic
concepts.

This approach has the added benefit of allowing algorithms to be
specialized based on refinement.  For example, most L/U
factorization algorithms involve pivoting steps designed to reduce
the magnitude of errors induced by floating-point operations.
However, when the element type models a precise algebraic
concept (e.g. an infinite-precision rational number type), those
pivoting steps are not required.  A similar effect occurs in
simulations where matrices with the same sparse structure are
factored repeatedly: in calculating the sparse structure of the
result, a boolean “fill” type that requires no pivoting can be used.

Andrew Lumsdaine notes (2005-1-28) that

  “Another simpler example of where things can be sped up in
  infinite precision case is in just adding up a list of numbers.
  To do this with high accuracy with floats you want to sort,
  normalize, etc.  With infinite precision, you can just add them
  up.”

and

  “We should probably also distinguish infinite precision from
  infinite length.  I.e., integers can be added without error, but
  not if they overflow.  So perhaps a Bounded concept as well.  A
  float therefore models :concept:`FinitePrecision` and
  :concept:`Bounded`\ ”

.. [#fieldwitherror] Pick a different name if you like.

.. _`preliminary documentation`: ../external/prototype_manual.pdf

.. [High02] `Nicholas J. Higham`_, *Accuracy and Stability of Numerical
   Algorithms, Second edition*, SIAM_, 2002, xxx+680 pp, ISBN
   0-89871-521-0.  http://www.ma.man.ac.uk/~higham/asna/

.. _`Nicholas J. Higham`: http://www.ma.man.ac.uk/~higham
.. _SIAM: http://www.siam.org/

Deep vs. Shallow Copy Semantics
-------------------------------

Unlike previous incarnations of MTL, we do *not* plan to use a
handle-body implementation for matrices and vectors.

* except for views and adapters, which explicitly do not own data,
  copy constructors should copy (no "handles").  Rationale: this
  models the well-understood behavior of mathematical primitives.
  Stack-based and heap-based objects have consistent behavior.  As
  an upshot of both these facts, there is less chance of confusing
  bugs.

* assignment operators should always copy.  Views and adapters copy
  over their target elements when assigned.  Rationale: ditto.

* Efficiency issues can be handled using library implementations of
  move semantics.  "Perfect" move semantics are possible in most
  modern compilers today, and with recent developments in the core
  working group that capability will become mandated
  (http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#291)
  and even automatic
  (http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#391).
  None of this was available when Jeremy wrote his paper.

* Issues of views and reference binding (see
  http://www.osl.iu.edu/research/mtl/reference/html/MTL_Object_Model.html)
  can be dealt with by returning const views from adapter
  functions.  For example::

     template <class MatrixType>
     const transpose_view<MatrixType> transpose(MatrixType& m);

  consider::

     typedef transpose_view<matrix<> > t;
     typedef transpose_view<matrix<> const> tc;

  The library supplies ``t`` with ``const`` member functions and
  free functions accepting ``t const&`` that can mutate ``t``\ 's
  referent matrix.

  The library only supplies ``tc`` with ``const`` member functions
  and free functions accepting ``tc const&`` that cannot mutate
  ``tc``\ 's referent matrix.

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


