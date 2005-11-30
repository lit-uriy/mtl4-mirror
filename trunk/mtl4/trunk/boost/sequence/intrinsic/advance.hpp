// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_ADVANCE_DWA20051128_HPP
# define BOOST_SEQUENCE_INTRINSIC_ADVANCE_DWA20051128_HPP

namespace boost { namespace sequence { namespace intrinsic { 

template <class Cursor, class Distance>
struct advance
{
    typedef Cursor type;
      
    type operator()(Cursor c, Distance const& d) const
    {
        return c += d;
    }
};
  
template <class Cursor, class Distance>
struct advance<Cursor const, Distance const>
  : advance<Cursor,Distance>
{};
  
template <class Cursor, class Distance>
struct advance<Cursor const, Distance>
  : advance<Cursor,Distance>
{};
  
template <class Cursor, class Distance>
struct advance<Cursor, Distance const>
  : advance<Cursor,Distance>
{};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_ADVANCE_DWA20051128_HPP
