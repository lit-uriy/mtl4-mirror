// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_TAG_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSIC_TAG_DWA2005616_HPP

# include <boost/sequence/intrinsic/tag_fwd.hpp>
# include <boost/sequence/intrinsic/iterator_range_tag.hpp>

//
// The tag<S> metafunction provides a dispatch tag for selecting
// implementations of intrinsic sequence operations (begin, end,
// elements).  
//

namespace boost { namespace sequence { namespace intrinsic {

// By default we expect every Sequence to be a model of
// SinglePassRange (see the Boost Range library documentation at
// http://www.boost.org/libs/range/doc/range.html).
template <class Sequence>
struct tag_impl
{
    typedef iterator_range_tag type;
};

template <class T>
struct tag
  : tag_impl<T>
{};

// The tag for T const is the same as that for T
template < class T >
struct tag< T const >
  : tag<T>
{};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_TAG_DWA2005616_HPP
