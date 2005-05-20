// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_ACCESSOR_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_ACCESSOR_DWA200555_HPP

# include <boost/sequence/fixed_size/is_fixed_size.hpp>
# include <boost/sequence/accessor_fwd.hpp>
# include <boost/sequence/index_property_map.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence {

namespace accessor_
{ 
  template <class Sequence>
  struct implementation<
      Sequence
    , typename enable_if<
          fixed_size::is_fixed_size<Sequence>
      >::type
  >
  {
      typedef index_property_map<Sequence> type;
  };
}

}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_ACCESSOR_DWA200555_HPP
