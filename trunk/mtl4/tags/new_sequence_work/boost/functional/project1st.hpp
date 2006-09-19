// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FUNCTIONAL_PROJECT1ST_DWA2006516_HPP
# define BOOST_FUNCTIONAL_PROJECT1ST_DWA2006516_HPP

namespace boost { namespace functional { 

struct project1st
{
    template <class Sig> struct result {};
    
    template <class F, class U>
    struct result<F(U&)>
    {
        typedef typename U::first_type& type;
    };

    template <class F, class U>
    struct result<F(U const&)>
    {
        typedef typename U::first_type const& type;
    };

    template <class F, class U>
    struct result<F(U)>
      : result<F(U const&)>
    {};

    template <class U>
    typename result<project1st const(U&)>::type
    operator()(U& x) const
    {
        return x.first;
    }
};

}} // namespace boost::functional

#endif // BOOST_FUNCTIONAL_PROJECT1ST_DWA2006516_HPP
