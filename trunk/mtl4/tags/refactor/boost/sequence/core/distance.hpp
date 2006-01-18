// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_DISTANCE_DWA20051128_HPP
# define BOOST_SEQUENCE_INTRINSIC_DISTANCE_DWA20051128_HPP

# include <boost/iterator/iterator_traits.hpp>

namespace boost { namespace sequence { namespace intrinsic { 

template <class Cursor1,class Cursor2>
struct distance;

template <class Cursor1>
struct distance<Cursor1,Cursor1>
{
    typedef typename iterator_difference<Cursor1>::type type;
      
    type operator()(Cursor1 const& c1, Cursor1 const& c2) const
    {
        return c2 - c1;
    }
};

template <class Cursor1, class Cursor2>
struct distance<Cursor1 const, Cursor2 const>
  : distance<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct distance<Cursor1 const, Cursor2>
  : distance<Cursor1,Cursor2>
{};
  
template <class Cursor1, class Cursor2>
struct distance<Cursor1, Cursor2 const>
  : distance<Cursor1,Cursor2>
{};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_DISTANCE_DWA20051128_HPP
