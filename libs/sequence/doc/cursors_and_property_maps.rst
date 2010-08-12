================================================
 Summary of the Cursor/Property Map Abstraction
================================================

:Authors: Dietmar Kühl and David Abrahams
:Date: April 14, 2005

.. contents:: Index

.. role:: concept
   :class: interpreted

:Abstract: We propose a new formulation for the sequence traversal
  and element access capabilities currently captured in the
  standard iterator concepts.

Problems with C++03 Iterators
-----------------------------

The C++03 iterator concepts couple access capabilities—such as
whether or not a dereferenced iterator must produce an lvalue—to
traversal capabilities such as whether the iterator can move
backwards.  Two of the problems caused by this coupling have been
extensively detailed in earlier work [ASW04]_, namely:

- Algorithm requirements are more strict than necessary, because
  they cannot separate the need for random access or bidirectional
  traversal from the need for a true reference return type.

- Iterators are often mis-categorized. Iterators that support
  random access traversal but use “proxy references” are classic
  examples.

However, the need for proxy references themselves could be seen as
a design flaw in the iterator concepts: the syntax for writing into
a dereferenced pointer complicates any writable iterator that
cannot easily supply an in-memory instance of the type being
written (consider an iterator over a disk-based container).

Achieving interoperability between a container's ``iterator`` and
its ``const_iterator`` is a well-known trouble spot for
implementors, and it was wrong in many early STL implementations.
The need for separate ``iterator`` and ``const_iterator`` types,
and redundant boilerplate ``begin``/\ ``end``/\ ``rbegin``/\
``rend`` overloads in containers is another consequence of the
coupling described earlier.

Finally, there is the issue of converting constant iterators to
mutable ones.  Consider::

  template <SomeContainer>
  class encapsulated
  {
   public:
      typedef typename SomeContainer::const_iterator iterator;

      iterator begin() const { return x.begin(); }
      iterator end() const { return x.end(); }

      void transmogrify(iterator start, iterator end);

   private:
      SomeContainer x;
  };

An ``encapsulated<list<int> >`` passes out only
``list<int>::const_iterator`` so that users don't make
invariant-breaking changes.  All mutation should happen within the
``transmogrify`` member function, where ``encapsulated<>`` can
manage the changes.  But ``encapsulated`` will have trouble
implementing any mutations inside its ``transmogrify`` member
function because the iterators supplied by the user—which ought to
be nothing more than position indicators—actually regulate access.

Proposed Solution
-----------------

We propose to solve these problems by dividing the responsibilities
of an iterator into two objects: a *cursor* and a *property map*.
A **cursor** is actually a kind of iterator, but instead of
iterating over sequence elements, it iterates over *keys*.  A
**key** is just a representation of a position in the sequence.  A
**property map** is just a kind of function object that associates
keys with values.  To read the value at a particular position, pass
its key to the property map.  To write a new value into a position,
pass the position's key and the new value to the property map.

.. |Series| replace:: :concept:`Series`

An ordinary iterator range is be represented by a property map and
two cursors, which form a |Series|.  The following table shows how
some iterator operations correspond to those on cursors and
property maps:

  ============= ================  ====================
  Operation     C++'98 approach     new approach
  ============= ================  ====================
  read          ``val = *it``     ``val = pm(*c)``
  write         ``*it = val``     ``pm(*c, val)``
  lvalue access ``X& a = *it``    ``X& a = pm(*c)``
  preincrement  ``++it``          ``++c``
  random offset ``it + n``        ``c + n``
  *...etc.*     
  ============= ================  ====================

In the table above, ``it`` is an iterator, and ``pm`` and ``c`` are
a corresponding property map and cursor.  ``X`` is the iterator's
value type and ``val`` is an object of type ``X``.

::

  concept<typename C, typename PM>
  where 
  { 
      ReadableIterator<C>
    , Callable1<PM, ReadableIterator<C>::value_type> 
  }
  ReadableSeries
  {
      typedef result_of<PM(C::value_type)>::type value_type;
  };

  concept<typename C, typename PM, typename V>
  where 
  { 
      ReadableIterator<C>
    , Callable2<PM, ReadableIterator<C>::value_type, V> 
  }
  WritableSeries;

.. table:: |Series|\ ``<typename C, typename PM>`` Requirements


Cursors are essentially input iterators, optionally
refined with bidirectional or random access traversal.  The cursor
concept is never refined to allow mutability or lvalue
access—those features are the sole domain of property maps—so
access is fully decoupled from traversal in the concept framework.

The presence of a dereference operator on cursors might seem
strange, since since the goal is to separate access from traversal.
For instance, the values of a property map could be read as
``pm(c)`` instead of ``pm(*c)``.  It turns out that the dereference
operation is required in order to reasonably support traversal
adapters.  If you have a property map that accepts ``foo_cursor``
arguments, it may not be designed to accept a
``reverse_cursor<foo_cursor>`` directly.  Like so many others, this
problem is solved by adding a level of indirection: dereferencing a
cursor yields an intermediate *key*, and the key type of a
traversal adapter is the same as that of its underlying iterator.

Backward Compatibility
----------------------

Raw pointers and, indeed, all C++98 iterators have a natural place
in this scheme.  Combined with an *identity* property map, an
iterator can act as a cursor:

.. _identity_property_map:

::

  // A property map whose keys and values are the same
  struct identity_property_map
  {
      // Readability
      template <class K>
      inline
      K& operator()(K& k) const
      {
          return k;
      }

      template <class K>
      inline
      K const& operator()(K const& k) const
      {
          return k;
      }

      // Writability
      template <class K, class V>
      inline
      void operator()(K& k, V const& v) const
      {
          return k = v;
      }

      // This one is needed to support proxies
      template <class K, class V>
      inline
      void operator()(K const& k, V const& v) const
      {
          return k = v;
      }
  };

C++98 algorithms can be extended to accept optional property maps,
with instances of ``identity_property_map`` as the default.

Associated Types
----------------

To access the key type of a cursor (the type returned when it is
dereferenced), we can use the ``key_type`` metafunction::

  typename key_type<Cursor>::type key = *c;

The (default) implementation of ``key_type`` is::

  template <class Cursor>
  struct key_type
  {
      typedef typename 
        std::iterator_traits<Cursor>::reference type;
  };

.. Note:: It's important that key_type be reference-preserving if
   ``identity_property_map`` is going to work with cursors that are
   iterators.  Consider

Property maps don't necessarily have a “value type.”  Indeed, the
``identity_property_map`` shown above can read and write arbitrary
types.  To discover the type accessed by a given key type ``K``
through a property map of type ``PropertyMap``, we can write::

   result_of<PropertyMap(K)>::type

In other words, due to its use of the function call interface, we
don't need to introduce a new trait metafunction to describe the
result of accessing a property map.

.. [ASW04] David Abrahams, Jeremy Siek, Thomas Witt, `New Iterator
   Concepts`,
   2004. http://www.boost.org/libs/iterator/doc/new-iter-concepts.html

