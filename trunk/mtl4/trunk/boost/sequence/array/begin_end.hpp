// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef PRIMITIVES_DWA2005330_HPP
# define PRIMITIVES_DWA2005330_HPP

# include <boost/sequence/fixed_size/cursor.hpp>

namespace boost {
namespace sequence {
namespace array
{ 
  using fixed_size::cursor;

  template <class T, std::size_t N>
  cursor<0>
  begin( T (&) [N] )
  {
      return cursor<0>();
  }
  
  template <class T, std::size_t N>
  cursor<N>
  end( T (&) [N] )
  {
      return cursor<N>();
  }

} // namespace array

template <class T, std::size_t N>
struct begin_cursor<T (&) [N]>
{
    typedef cursor<0> type;
}

template <class T, std::size_t N>
struct end_cursor<T (&) [N]>
{
    typedef cursor<N> type;
}

template <class T, std::size_t N>
struct begin_cursor<T[N]>
{
    typedef cursor<0> type;
}

template <class T, std::size_t N>
struct end_cursor<T[N]>
{
    typedef cursor<N> type;
}

}} // namespace boost::sequence

#endif // PRIMITIVES_DWA2005330_HPP
