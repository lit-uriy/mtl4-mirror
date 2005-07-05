// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
# define BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP

# include <boost/sequence/detail/config.hpp>

namespace boost { namespace sequence { namespace detail { 

template <class T, int = 0>
struct instance
{
    static T& get()
    {
        static T x;
        return x;
    }
};

# if BOOST_WORKAROUND(BOOST_GNUC_FULL_VERSION, <= 3003003)
#  define BOOST_SEQUENCE_DECLARE_INSTANCE(type, name) namespace { type name; }
# else
#  define BOOST_SEQUENCE_DECLARE_INSTANCE(type, name) namespace { type const& name = sequence::detail::instance< type >::get(); }
# endif

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_INSTANCE_DWA200559_HPP
