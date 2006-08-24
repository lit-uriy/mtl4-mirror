// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FUNCTIONAL_LESS_DWA2006519_HPP
# define BOOST_FUNCTIONAL_LESS_DWA2006519_HPP

# include <functional>

namespace boost { namespace functional { 

struct less
{
    typedef bool result_type;
    
    template <class T>
    bool operator()(T const& x, T const& y) const
    {
        return std::less<T>()(x,y);
    }

    template <class T, class U>
    bool operator()(T const& x, U const& y) const
    {
        return x < y;
    }
};

}} // namespace boost::functional

#endif // BOOST_FUNCTIONAL_LESS_DWA2006519_HPP
