// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef COPY_DWA200554_HPP
# define COPY_DWA200554_HPP

# include <boost/sequence/algorithm/dispatch.hpp>
# include <boost/sequence/detail/instance.hpp>
# include <boost/sequence/algorithm/fixed_size/copy.hpp>
# include <boost/typeof/typeof.hpp>
# include <boost/type_traits/add_const.hpp>

# include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace boost { namespace sequence { namespace algorithm {

struct copy_
{
    // The use of add_const below is needed to work around a VC7.1 bug
    template <class Range1, class Range2>
    typename dispatch<copy_(typename add_const<Range1>::type&,Range2&)>::type
    operator()(Range1 const& src, Range2& dst) const
    {
# if BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
        return
            typename dispatch<copy_(Range1 const&,Range2&)>::implementation
            ().execute(src,dst);
# else
        return dispatch<copy_(Range1 const&,Range2&)>::implementation::execute(src,dst);
# endif 
    }
};

copy_ const& copy = detail::instance<copy_>::object;

}}} // namespace boost::sequence::algorithm

BOOST_TYPEOF_REGISTER_TYPE(boost::sequence::algorithm::copy_);

#endif // COPY_DWA200554_HPP
