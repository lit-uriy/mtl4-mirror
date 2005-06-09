================================================
 Summary of the Cursor/Property Map Abstraction
================================================

:Authors: Dietmar Kühl and David Abrahams
:Date: April 14, 2005

The C++03 iterator concepts couple access capabilities—such as
whether or not a dereferenced iterator must produce an lvalue—to
traversal capabilities such as whether the iterator can move
backwards.  This concept hierarchy fails to capture the properties
many of useful iterators, for example, those of vector<bool>.  In
addition, the syntax for writing into a dereferenced pointer
complicates the implementation of some iterators by forcing the
introduction of proxy references.

We propose to solve these problems by dividing the responsibilities
of iterators into two objects: cursors that traverse a sequence of
keys, and property maps that associate those keys with values.  An
ordinary iterator range is be represented by a property map and
two cursors.  The following table shows how some iterator
operations correspond to those on cursors and property maps:

  ============= ================  ====================
  Operation     C++'98 approach     new approach
  ============= ================  ====================
  read          ``val = *it``     ``val = pm(*c)``
  write         ``*it = val``     ``pm(*c, val)``
  lvalue        ``X& a = *it``    ``X& a = pm(*c)``
  preincrement  ``++it``          ``++c``
  random offset ``it + n``        ``c + n``
  *...etc.*     
  ============= ================  ====================

In the table above, ``it`` is an iterator, and ``pm`` and ``c`` are
a corresponding property map and cursor.  ``X`` is the property
map's value type and ``val`` is an object of type ``X``.

Cursors are essentially read-only (input) iterators, optionally
refined with bidirectional or random access traversal.  The cursor
concept is never refined to allow writability or lvalue
access—those features are the sole domain of property maps—so
access is fully decoupled from traversal in the concept framework.

You might ask, “why not just access the values of a property map as
``pm(c)`` instead of ``pm(*c)?``\ ” The reason that cursors need to
be “dereferenced” is that you will want to build traversal adapters
for them.  If you have a property map that accepts ``foo_cursor``
arguments, it probably won't accept a
``reverse_cursor<foo_cursor>`` directly.  Dereferencing a cursor is
just an operation that gets you to a common key type that can be
used to index the property map.

Cursors get you out of the whole, nasty iterator/const_iterator
debacle.  Access privileges are firmly lodged with the property map,
where they should be.

Cursors get you out of the whole nasty “iterator unwrapping” debacle.
If you want to search through a list of pairs for an item whose
``first`` element matches some predicate and then use that position as
the beginning of a view onto the ``second`` elements, you can use the
same exact cursor in both cases.  It is a pure position.  You just
property maps that access the first/second elements respectively.

Property maps don't necessarily need to have a "value_type".  They
do have ::

   result_of<pm(std::iterator_traits<Cursor>::reference)>::type

in other words, we don't need to proliferate another ugly traits
metafunction.

