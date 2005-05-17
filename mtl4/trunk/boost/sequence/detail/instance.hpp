// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
# define BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP

namespace boost { namespace sequence { namespace detail { 

template <class T, int = 0>
struct instance
{
    static T object;
};

template <class T, int x>
T instance<T,x>::object;

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
