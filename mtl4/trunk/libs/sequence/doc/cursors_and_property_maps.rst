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
of iterators into two objects: cursors, which traverse a sequence
of keys, and property maps, which associate those keys with values.
An ordinary iterator range is be represented by a property map and
two cursors.  The following table shows how some iterator
operations correspond to those on cursors and property maps:

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
map's value type and ``val`` is an object of type ``X``.

Cursors are essentially input iterators, optionally
refined with bidirectional or random access traversal.  The cursor
concept is never refined to allow mutability or lvalue
access—those features are the sole domain of property maps—so
access is fully decoupled from traversal in the concept framework.

If we are intent on separating access from traversal, it seem
natural to remove the use of dereference operators from cursors
completely, so the values of a property map would be read as
``pm(c)`` instead of ``pm(*c)``.  It turns out that the dereference
operation is required in order to reasonably support traversal
adapters.  If you have a property map that accepts ``foo_cursor``
arguments, it may not be designed to accept a
``reverse_cursor<foo_cursor>`` directly.  Dereferencing yields a
*key*: a value of a common type that can be used to access the
property map no matter what traversal pattern is being used.

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

An obvious [#obvious]_ default implementation for ``key_type`` is::

  template <class Cursor>
  struct key_type
  {
      typedef typename 
        std::iterator_traits<Cursor>::value_type type;
  };

Property maps don't necessarily have a “value type.”  Indeed, the
``identity_property_map`` shown above can read and write arbitrary
types.  To discover the type accessed by a given key type ``K``
through a property map of type ``PropertyMap``, we can write::

   result_of<PropertyMap(Key)>::type

In other words, due to its use of the function call interface, we
don't need to introduce a new trait metafunction to describe the
result of accessing a property map.

.. [#obvious] It isn't clear yet whether it would be more useful to
   know when the key type is an lvalue.  In that case, ::

      template <class Cursor>
      struct key_type
      {
          typedef typename 
            std::iterator_traits<Cursor>::reference type;
      };

   might be a more appropriate implementation.

.. [ASW04] David Abrahams, Jeremy Siek, Thomas Witt, `New Iterator
   Concepts`,
   2004. http://www.boost.org/libs/iterator/doc/new-iter-concepts.html

