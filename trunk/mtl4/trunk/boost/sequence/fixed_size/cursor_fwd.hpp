// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_CURSOR_FWD_DWA200562_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_CURSOR_FWD_DWA200562_HPP

# include <cstddef>

namespace boost { namespace sequence { namespace fixed_size { 

namespace cursor_ // namespace for ADL protection.
{
  template <std::size_t N> struct cursor;
}

using cursor_::cursor;
    
}}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_CURSOR_FWD_DWA200562_HPP
