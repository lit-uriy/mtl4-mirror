==================================================
 Non-Intrusive Dispatching for Generic Algorithms
==================================================

.. sectnum::

:abstract:

   The Fixed Algorithm Size Template (FAST) library is a library of
   STL-like algorithm implementations for operating on sequences
   whose size is known at compile-time.  In considering its
   interface, we realized that similar libraries would soon be
   developed for other kinds of sequences such as those described
   in [Austern98]_ (notably for distributed data).  In fact, it
   became clear that we would want to use both kinds of
   sequences (in addition to traditional STL sequences) generically
   in the implementation of linear algebra components.  We describe
   how disparate algorithm implementations can be invoked with a
   single calling syntax in C++.  We also reveal some important
   principles of generic programming that have heretofore received
   little attention, and are not well-addressed by C++.

----------------------
Description of Problem
----------------------

.. role:: concept
   :class: interpreted

In the C++ standard library, algorithm implementations are selected
from a limited number of possibilities based on the concepts
modeled by the iterators passed.  For example, take this
``find_end`` algorithm implementation from a recent version of
libstdc++::

  template<typename _ForwardIter1, typename _ForwardIter2>
    _ForwardIter1
    find_end(_ForwardIter1 __first1, _ForwardIter1 __last1,
	     _ForwardIter2 __first2, _ForwardIter2 __last2)
    {
      return __find_end(__first1, __last1, __first2, __last2,
			__iterator_category(__first1),
			__iterator_category(__first2));
    }

One of these two implementations of ``__find_end`` is selected,
based on the most-refined common category of the iterators::

  // find_end for forward iterators.
  template<typename _ForwardIter1, typename _ForwardIter2>
    _ForwardIter1
    __find_end(_ForwardIter1 __first1, _ForwardIter1 __last1,
	       _ForwardIter2 __first2, _ForwardIter2 __last2,
	       forward_iterator_tag, forward_iterator_tag)
    {
       ...
    }

  // find_end for bidirectional iterators.  
  template<typename _BidirectionalIter1, typename _BidirectionalIter2>
    _BidirectionalIter1
    __find_end(_BidirectionalIter1 __first1, _BidirectionalIter1 __last1,
	       _BidirectionalIter2 __first2, _BidirectionalIter2 __last2,
	       bidirectional_iterator_tag, bidirectional_iterator_tag)
    {
       ...
    }

For the purposes of this discussion, we are building algorithms
that operate on :concept:`Ranges` that package a start and an end
iterator in a single object.

-----------------------------
Aspects of the Ideal Solution
-----------------------------

1. Algorithm implementations should be decoupled from one another.
   It should be possible to add new kinds of sequences and related
   algorithm implementations without modifying existing code.

2. Algorithm Implementations should be decoupled from specific
   sequence types.

3. 

4. Ambiguities between "equally good" implementations should be
   resolved automatically, so that users don't have to manage
   errors in library code.

------------------------------
Approaching the Problem in C++ 
------------------------------

.. Note:: There's lots left to write here.

-----------------
What we've Gained
-----------------

- Argument dependent lookup is restricted to one name

- The namespace of an algorithm implementation is not tied to the
  namespaces of its arguments


.. nothing yet

.. [Austern98] Austern, Matt. *Segmented Iterators and Hierarchical
   Algorithms*. http://lafstern.org/matt/segmented.pdf  from
   M. Jazayeri, R. Loos, D. Musser (eds.): Generic Programming,
   Proc. of a Dagstuhl Seminar, Lecture Notes on Computer
   Science. Volume. 1766  
