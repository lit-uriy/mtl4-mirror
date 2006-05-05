// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_OPERATIONS_NEXT_JDG20060207_HPP
# define BOOST_SEQUENCE_OPERATIONS_NEXT_JDG20060207_HPP

# include <boost/sequence/operations/iterator_range_operations.hpp>

namespace boost { namespace sequence { namespace intrinsic {

template <class Cursor>
struct next
{
    typedef Cursor type;
    Cursor operator()(Cursor x) const
    {
        return ++x;
    }
};

template <class Cursor>
struct next<Cursor const>
  : next<Cursor>
{};
    
}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_OPERATIONS_NEXT_JDG20060207_HPP
