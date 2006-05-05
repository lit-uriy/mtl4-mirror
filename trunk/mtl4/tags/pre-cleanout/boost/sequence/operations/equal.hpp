// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_OPERATIONS_EQUAL_JDG20060207_HPP
# define BOOST_SEQUENCE_OPERATIONS_EQUAL_JDG20060207_HPP

# include <boost/sequence/operations/iterator_range_operations.hpp>

namespace boost { namespace sequence { namespace intrinsic {

template <class Cursor1, class Cursor2>
struct equal
{
    typedef bool type;
      
    type operator()(Cursor1 const& c1, Cursor2 const& c2) const
    {
        return c1 == c2;
    }
};

template <class Cursor1, class Cursor2>
struct equal<Cursor1 const, Cursor2 const>
  : equal<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct equal<Cursor1 const, Cursor2>
  : equal<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct equal<Cursor1, Cursor2 const>
  : equal<Cursor1,Cursor2>
{};
  
}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_OPERATIONS_EQUAL_JDG20060207_HPP
