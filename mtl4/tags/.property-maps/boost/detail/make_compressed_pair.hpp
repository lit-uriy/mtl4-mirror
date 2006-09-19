// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_DETAIL_MAKE_COMPRESSED_PAIR_DWA2005511_HPP
# define BOOST_DETAIL_MAKE_COMPRESSED_PAIR_DWA2005511_HPP

# include <boost/compressed_pair.hpp>

namespace boost { namespace detail { 

template <class T1, class T2>
compressed_pair<T1,T2>
make_compressed_pair(T1 const& x1, T2 const& x2)
{
    return compressed_pair<T1,T2>(x1,x2);
}

}} // namespace boost::detail

#endif // BOOST_DETAIL_MAKE_COMPRESSED_PAIR_DWA2005511_HPP
