// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP

# include <boost/sequence/intrinsics_fwd.hpp>
# include <boost/sequence/fixed_size/intrinsics.hpp>
# include <boost/sequence/identity_property_map.hpp>
# include <boost/sequence/intrinsic/iterator_range_tag.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/const_iterator.hpp>

# include <boost/type_traits/is_array.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence { intrinsic {

// The implementation of the intrinsics specialization that applies to
// models of SinglePassRange (see the Boost Range library
// documentation at http://www.boost.org/libs/range/doc/range.html).
// The first argument is the range type and the second argument is a
// nullary metafunction that returns the iterator type to use as the
// Range's cursor.
//
// Note: the need for the 2nd argument may disappear when the Range
// library is changed so that range_iterator<R const>::type is the
// same as range_const_iterator<R>::type.
template <class Sequence, class GetIterator>
struct iterator_range_intrinsics
{
    struct begin
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::begin(s);
        }
    };
        
    struct end
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::end(s);
        }
    };
        
    struct elements
    {

        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return type();
        }
    };
};

// Intrinsics specializations for iterator ranges.
template <class Sequence>
struct intrinsics<Sequence, intrinsic::iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_iterator<Sequence>
    >
{};

template <class Sequence>
struct intrinsics<Sequence const, intrinsic::iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_const_iterator<Sequence>
    >
{};

namespace intrinsic
{
  // The default implementation of each intrinsic function object type
  // is inherited from the corresponding member of
  // intrinsics<Sequence>.  You can of course specialize begin<S>,
  // end<S>, and elements<S>, individually, but specializing
  // intrinsics<> usually more convenient.
  template <class Sequence>
  struct begin : intrinsics<Sequence>::begin {};
  
  template <class Sequence>
  struct end : intrinsics<Sequence>::end {};
  
  template <class Sequence>
  struct elements : intrinsics<Sequence>::elements {};

  // Specializations of function<Op> provide the actual type of each
  // intrinsic function object: Overloaded function call operators
  // handle rvalue binding without requiring boilerplate in
  // specializations of begin/end/elements.
  template <template <class> class Operation>
  struct function
  {
      template <class Sequence>
      typename Operation<Sequence>::type
      operator()(Sequence& s) const
      {
          return Operation<Sequence>()(s);
      }
      
      template <class Sequence>
      typename
      // VC-8.0 beta likes to match this overload even to non-const arrays
# if !BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050601))
        Operation<Sequence const>
# else 
        lazy_disable_if<is_array<Sequence>, Operation<Sequence const> >
# endif
      ::type
      operator()(Sequence const& s) const
      {
          return Operation<Sequence const>()(s);
      }
  };
  
}
    
}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_OPERATIONS_DWA2005616_HPP
