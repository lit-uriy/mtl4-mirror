.. Copyright David Abrahams 2005. Distributed under the Boost
.. Software License, Version 1.0. (See accompanying
.. file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

========================================
 To-Be-Proposed Boost Sequences Library
========================================

:abstract: This library under development is a re-think of the
  ideas we've used to represent sequences in C++: containers,
  iterators, and Ranges__.

__ http://www.boost.org/libs/range/

* About `dispatching intrinsic operations`_ like ``begin`` and
  ``end``.

.. _`dispatching intrinsic operations`: intrinsics.html

* About the `Cursor and Property Map Abstractions`_ that underlie
  `the Sequence concept`_.

.. _`Cursor and Property Map Abstractions`: cursors_and_property_maps.html

.. _`Sequence`:

--------------------
The Sequence Concept
--------------------

I am bundling two cursors and a property map into a concept called
“Sequence.”  I realize the standard has already used that term, but
there are few reasonable terms left and the standard's concept is
weak and seldom used.