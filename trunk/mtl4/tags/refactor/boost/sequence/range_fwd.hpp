// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_RANGE_FWD_DWA200562_HPP
# define BOOST_SEQUENCE_RANGE_FWD_DWA200562_HPP

namespace boost { namespace sequence { 

namespace range_
{
  struct not_stored;
  
  template <class Elements, class Begin, class End, class Size = not_stored>
  class range;
}

using range_::range;

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_RANGE_FWD_DWA200562_HPP
