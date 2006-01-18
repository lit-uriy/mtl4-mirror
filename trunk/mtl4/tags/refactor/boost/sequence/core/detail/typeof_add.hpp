// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_TYPEOF_ADD_DWA2005520_HPP
# define BOOST_SEQUENCE_DETAIL_TYPEOF_ADD_DWA2005520_HPP

# include <boost/typeof/typeof.hpp>
# include <boost/sequence/core/detail/make.hpp>

namespace boost { namespace sequence { namespace detail { 

template <class T, class U, bool workaround = false>
struct typeof_add;


template <class T, class U>
struct typeof_add<T,U,false>
{
    typedef BOOST_TYPEOF_TPL(make<T>() + make<U>()) type;
};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_TYPEOF_ADD_DWA2005520_HPP
