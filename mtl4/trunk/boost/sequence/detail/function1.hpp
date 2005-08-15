// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_FUNCTION1_DWA2005812_HPP
# define BOOST_SEQUENCE_DETAIL_FUNCTION1_DWA2005812_HPP

# include <boost/utility/enable_if.hpp>
# include <boost/sequence/detail/intrinsic_arg.hpp>
# include <boost/sequence/detail/msvc_array_wknd.hpp>

namespace boost { namespace sequence { namespace detail { 

// This facade "handles" the forwarding problem for its
// implementation, F<A>, a stateless function object type.
template <template <class> class F>
struct function1
{
    template <class X0>
    struct result
      : F<typename intrinsic_arg<X0>::type>
    {};

    template <class X0>
    BOOST_SEQUENCE_MSVC_ARRAY_WKND( (X0) , result<X0 const&> )
    operator()(X0 const& a0) const
    {
        return F<X0 const>()(a0);
    }

    template <class X0>
    typename result<X0&>::type
    operator()(X0& a0) const
    {
        return F<X0>()(a0);
    }
};

template <template <class> class F>
struct const_function1
{
    template <class A0>
    struct result
      : F<typename intrinsic_const_arg<A0>::type>
    {};

    template <class A0>
    typename F<A0>::type
    operator()(A0 const& a0) const
    {
        return F<A0>()(a0);
    }
};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_FUNCTION1_DWA2005812_HPP
