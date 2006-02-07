// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_OPERATIONS_ITERATOR_RANGE_OPERATIONS_JDG20060207_HPP
# define BOOST_SEQUENCE_OPERATIONS_ITERATOR_RANGE_OPERATIONS_JDG20060207_HPP

# include <boost/sequence/operations/operations_fwd.hpp>

# include <boost/sequence/core/property_map/identity_property_map.hpp>
# include <boost/sequence/core/iterator_range_tag.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/size.hpp>

# include <boost/type_traits/remove_cv.hpp>
# include <boost/type_traits/is_convertible.hpp>

namespace boost { namespace sequence { namespace intrinsic {

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
struct iterator_range_operations
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
        typedef identity_property_map type;
        
        type operator()(Sequence const&) const
        {
            return type();
        }
    };

    struct size
    {
        typedef typename boost::range_size<
            typename remove_cv<Sequence>::type
        >::type type;

        type operator()(Sequence& s) const
        {
            
            return boost::size(s);
        }
    };
};

// Intrinsics specializations for iterator ranges.
template <class Sequence>
struct operations<Sequence, iterator_range_tag>
  : iterator_range_operations<
# if 1
        Sequence, range_iterator<Sequence>
# else
        // This appears to be code for proxy support.  We will need
        // something like this, unless Thorsten fixes his library
        // first.
        typename mpl::if_<
            mpl::or_<
                is_convertible<
                    typename iterator_category<
                        typename range_iterator<Sequence>::type
                    >::type
                  , std::output_iterator_tag
                >
              , is_non_const_lvalue_iterator<
                    typename range_iterator<Sequence>::type
                >
            >
          , Sequence
          , Sequence const
        >::type
      , range_result_iterator<Sequence>
# endif 
    >
{};

template <class Sequence>
struct operations<Sequence const, intrinsic::iterator_range_tag>
  : iterator_range_operations<
        Sequence const, range_const_iterator<Sequence>
    >
{};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_ITERATOR_RANGE_OPERATIONS_JDG20060207_HPP
