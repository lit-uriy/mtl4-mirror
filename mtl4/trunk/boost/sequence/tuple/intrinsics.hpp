// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_TUPLE_INTRINSICS_DWA2005616_HPP
# define BOOST_SEQUENCE_TUPLE_INTRINSICS_DWA2005616_HPP

# include <boost/sequence/index_property_map.hpp>
# include <boost/sequence/intrinsic/operations_fwd.hpp>
# include <boost/sequence/tuple/cursor.hpp>
# include <boost/sequence/range.hpp>
# include <boost/range/result_iterator.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence {

namespace intrinsic
{
  // Provide the implementation of intrinsics for fixed-size sequence
  // types.
  template <class Sequence, std::size_t N>
  struct operations<Sequence, tuple::tag<N> >
  {
      struct begin
      {
          typedef tuple::cursor<0> type;
          type operator()(Sequence const&) const
          { return type(); }
      };
    
      struct end
      {
          typedef tuple::cursor<N> type;
          type operator()(Sequence const&) const
          { return type(); }
      };

      struct elements
      {
          typedef index_property_map<Sequence&> type;
          type operator()(Sequence& s) const
          { return type(s); }
      };
  };
}

}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_TUPLE_INTRINSICS_DWA2005616_HPP
