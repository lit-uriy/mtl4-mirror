// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef COPY_DWA200554_HPP
# define COPY_DWA200554_HPP

# include <boost/sequence/algorithm/dispatch.hpp>
# include <boost/sequence/detail/instance.hpp>
# include <boost/sequence/algorithm/fixed_size/unrolled.hpp>
# include <boost/typeof/typeof.hpp>

namespace boost { namespace sequence { namespace algorithm {

struct copy_
{
    template <class Range1, class Range2>
    typename dispatch<copy_(Range1,Range2)>::result
    operator()(Range1 const& src, Range2& dst) const
    {
        return dispatch<copy_(Range1 const&,Range2&)>::implementation
            ::execute(src,dst);
    }
};
BOOST_TYPEOF_REGISTER_TYPE(copy_);

copy_ const& copy = detail::instance<copy_>::object;

}}} // namespace boost::sequence::algorithm

#endif // COPY_DWA200554_HPP
