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

.. role:: concept
   :class: interpreted

.. |Sequence| replace:: :concept:`Sequence`
.. |Property Map| replace:: :concept:`Property Map`
.. |Writable Property Map| replace:: :concept:`Writable Property Map`
.. |Cursor| replace:: :concept:`Cursor`

.. _Cursor: cursors_and_property_maps.html
.. _Property Map: cursors_and_property_maps.html

------------------------
 The |Sequence| Concept
------------------------


|Sequence| is an abstraction that bundles two |Cursor|_\ s and a
|Property Map|_. [#naming]_ In the table below, ``S`` is a model of
|Sequence| and ``x`` is a (possibly *cv*\ -qualified) instance of
``S``.

.. table:: Sequence Requirements

   +------------------------------+----------------------------------------+---------------------+
   |Valid Expression              |Type                                    |Semantics            |
   +==============================+========================================+=====================+
   |::                            |::                                      |Returns a |Property  |
   |                              |                                        |Map|_ that accesses  |
   |  boost::sequence::elements(s)|  boost::result_of<                     |the elements of      |
   |                              |      boost::sequence::end::id(S const&)|``x``.  If ``x`` is  |
   |                              |  >::type                               |writable, the result |
   |                              |                                        |is a model of        |
   |                              |                                        ||Writable Property   |
   |                              |                                        |Map|.                |
   +------------------------------+----------------------------------------+---------------------+
   |``boost::sequence::begin(x)`` |::                                      |Returns a |Cursor|_  |
   |                              |                                        |that, when used with |
   |                              |  boost::result_of<                     |the ``elements``     |
   |                              |    boost::sequence::begin::id(S const&)|property map,        |
   |                              |  >::type                               |traverses the        |
   |                              |                                        |elements of ``x``.   |
   +------------------------------+----------------------------------------+---------------------+
   |``boost::sequence::end(x)``   |::                                      |Returns a suitble    |
   |                              |                                        ||Cursor|_ for        |
   |                              |  boost::result_of<                     |terminating the      |
   |                              |    boost::sequence::end::id(S const&)  |traversal of ``x``.  |
   |                              |  >::type                               |                     |
   +------------------------------+----------------------------------------+---------------------+

.. [#naming] We realize the standard has already used the term
   “sequence,” but there are few reasonable terms left and the
   standard's concept is weak and seldom used.