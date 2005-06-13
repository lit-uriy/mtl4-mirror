Sequence Intrinsics and Dispatching
===================================

:abstract: We discuss how the
  intrinsic accessors of the Sequence concept can be used and customized,
  i.e. how do we access (and specify, if we're writing a new Sequence)

  - the begin cursor
  - the type of the begin cursor
  - the end cursor
  - the type of the end cursor
  - the property map
  - the type of the property map

.. contents:: Index

Known kinds of sequences
------------------------

  - boost::array<T,N>
  - T[N]
  - Standard containers
  - range<pmap,start,finish>
  - cv qualifications and references thereto

Fixed-Size Random Access Sequences
----------------------------------

- ``boost::array<T,N>``
- ``T[N]``
- ``range<P,fixed_size::cursor<N1>,fixed_size::cursor<N2> >``

Standard Containers and Default Behavior
----------------------------------------

Delegating to Boost.Range should be the default for most purposes.  It
already supplies ``begin()`` and ``end()`` functions that get you iterators
into something satisfying `Single Pass Range`_.

The cursor type should be whatever iterator Boost.Range gives us.  The
property map would be an ``identity_property_map`` that simply
returns whatever key it is passed, in this case, the iterator's
``reference`` type.

.. _`Single Pass Range`: http://www.boost.org/libs/range/doc/range.html#single_pass_range

Dispatching Issues
------------------

Typically arguments to intrinsics fall into three categories:

1. Types supplied by the library for which it can also supply
   behavior

2. Types designed by users to work with the library, for which the
   type's author supplies behavior

3. Built-in types and types that come from “third party”
   namespaces, such as ``T[N]`` and ``boost::array<T,N>``.

Those in category 3 need to be adapted in order to satisfy the
sequence concept directly.  It's not very clear that there will be
many types in category 2 at all.  I think most sequences will
probably match the Range concept from Boost.Range already.

The usual dispatching tradeoffs apply:

a. ADL can't work for category 3, unless you allow people to
   overload in namespaces they don't own, and we don't want to do
   that.

b. ADL is the only scheme that trivially allows any class derived
   from a model of Sequence to also model Sequence.

c. You can get some of the way there by using tag dispatching.
   Types in categories 1 and 2 can have nested tag types that are
   quite naturally inherited by derived classes.

d. Using specialization users can customize for any type or tag.
   It also gives them a convenient place to put their
   metafunctions.

I don't think consideration b is important in this case.  Therefore
I am left with specialization with the option to use tag
dispatching.

Customization Interface
-----------------------

The plain specialization interface would look like::

  namespace boost { namespace sequence { intrinsic {

  template <class T>
  struct begin<my_sequence<T> >
  {
      typedef some_begin_cursor type;  // should this be
                                       // result_type, for result_of?

      type operator()(my_sequence<T> const&) const;
  };

  }}}

The interface using specialization with tag dispatching might look
like::

  namespace boost { namespace sequence { intrinsic {

  struct my_sequence_tag {};

  template <class T>
  struct tag<my_sequence<T> >
  {
      typedef my_sequence_tag {};
  };

  template <>
  struct begin<tag>
  {
      typedef some_begin_cursor type;  // should this be
                                       // result_type, for result_of?

      type operator()(my_sequence<T> const&) const;
  };

  }}}

In order to simplify things, we could supply a default ``tag<S>``
implementation::

  namespace boost { namespace sequence { intrinsic {

  template <class S>
  struct tag { typedef S type; };

  }}}

That would mean you could use the plain specialization interface
when that is more appropriate, because a type's tag would be the
type itself, by default.  However, at this point it seems like a
needless generalization.  If we find a use case for it, we can
always change things in a backwards-compatible way; in the meantime
the plain specialization interface will be fine.

Functions vs. Objects
---------------------

I find very appealing the FC++/Phoenix approach of providing
callable objects instead of functions or function templates,
because it allows compile-time polymorphic functions to be passed
to other higer-order functions:

.. parsed-literal::

  namespace boost { namespace sequence {

  struct begin_function
    : provide_result [#provide_result]_\ <begin_function, intrinsic::begin<_> >
  {
      template <class Sequence>
      typename intrinsic::begin<Sequence>::type
      operator()(Sequence const& s) const
      {
          return intrinsic::begin<Sequence>()(s);
      }
  };

  namespace 
  {
    // begin in every TU will refer to the same object, and you
    // can't take the address of a reference, so this is safe in a
    // header file.
    begin_function const& begin =
      detail::instance<begin_function>::object;
  }

  }}

.. [#provide_result] The ``provide_result`` base class template
   gives us compatibility with ``tr1::result_of`` and will generate
   something like this::

      // tr1::result_of will check
      //
      //   begin_function::result<
      //       begin_function(Sequence const&)
      //   >::type 
      //
      // This sort of facility can be provided by a base class
      // template 
      template <class Signature> 
      struct result
        : typename intrinsic::begin<
              typename remove_const_ref<
                  first_argument_type<T>::type
              >::type
          >
      {};

User Interface
--------------

The customization interface described above is convenient for
sequence authors because it groups compile-time return type
computation with the runtime result computation.  However, users of
sequences will not want to name a class template specialization
such as ``begin<S>`` in order to call begin.  Something like ::

  boost::sequence::begin(s)

would be more appropriate.  But how should users ask for the begin
iterator type? ::

  boost::sequence::begin<S>::type

is impossible.  According to the customization interface described
above, we would use ::

  boost::sequence::intrinsic::begin<S>::type

but I am not particularly fond of the name ``intrinsic`` for this
purpose.  A namespace alias can allow us to pick something else,
but what would be better?

  boost::sequence::types::begin<S>::type
  boost::sequence::typeof::begin<S>::type
  boost::sequence::result_of::begin<S>::type

Another possibility is ::

  result_of< begin_ ( S const& ) >::type

Why Distinguish Homogeneity in Fixed Size Sequences
---------------------------------------------------

The importance of homogeneity, at least for copy, is in reduced
template instantiations.  If the sequence is homogeneous, a cursor can
be converted to a single runtime type and you can describe an
algorithm in terms of runtime positions and compile-time lengths,
which reduces the number of template instantiations to O(`log N`).

This implies we need a way to convert the cursor.  Maybe just ``c + 1``.





