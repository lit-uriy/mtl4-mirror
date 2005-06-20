// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP

# include <boost/sequence/tag_fwd.hpp>

# include <boost/type_traits/is_array.hpp>

namespace boost { namespace sequence { 

template <class Sequence, class Tag = typename tag<Sequence>::type>
struct intrinsics;

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP
