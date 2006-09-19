.. Copyright David Abrahams 2006. Distributed under the Boost
.. Software License, Version 1.0. (See accompanying
.. file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Cursors
=======

A cursor is a generalized concept that encompasses STL iterators and Fusion
tuple iterators.

=================== ============= ==============
Sequence Type       Element Types Cursor Types
=================== ============= ==============
STL container       Homogeneous   Homogeneous
Fixed-sized array   Homogeneous   Heterogeneous
Fusion tuple        Heterogeneous Heterogeneous
=================== ============= ==============

The most interesting case is a cursor over a fixed-sized
array. Although the elements are homogeneous, cursors are
heterogeneous -- they denote position at compile time.