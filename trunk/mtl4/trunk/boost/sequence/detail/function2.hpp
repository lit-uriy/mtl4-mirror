// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_FUNCTION2_DWA2005812_HPP
# define BOOST_SEQUENCE_DETAIL_FUNCTION2_DWA2005812_HPP

# include <boost/sequence/detail/intrinsic_arg.hpp>
# include <boost/mpl/or.hpp>
# include <boost/sequence/detail/msvc_array_wknd.hpp>
# include <boost/sequence/detail/comma_protect.hpp>

namespace boost { namespace sequence { namespace detail { 

// This facade "handles" the forwarding problem for its
// implementation, F, a stateless binary function object type.
template <template <class, class> class F>
struct function2
{
    template <class A0, class A1>
    struct result
      : F<
            typename intrinsic_arg<A0>::type
          , typename intrinsic_arg<A1>::type
        >
    {};

    
    template <class A0, class A1>
    BOOST_SEQUENCE_MSVC_ARRAY_WKND((A0)(A1), (result<A0 const&, A1 const&>))
    operator()(A0 const& a0, A1 const& a1) const
    {
        return F<A0 const,A1 const>()(a0, a1);
    }

    template <class A0, class A1>
    BOOST_SEQUENCE_MSVC_ARRAY_WKND((A1), (result<A0&, A1 const&>))
    operator()(A0& a0, A1 const& a1) const
    {
        return F<A0,A1 const>()(a0, a1);
    }

    template <class A0, class A1>
    BOOST_SEQUENCE_MSVC_ARRAY_WKND((A0), (result<A0 const&, A1&>))
    operator()(A0 const& a0, A1& a1) const
    {
        return F<A0 const,A1>()(a0, a1);
    }

    template <class A0, class A1>
    typename result<A0&, A1&>::type
    operator()(A0& a0, A1& a1) const
    {
        return F<A0,A1>()(a0, a1);
    }
};


template <template <class, class> class F>
struct const_function2
{
    template <class A0, class A1>
    struct result
      : F<
            typename intrinsic_const_arg<A0>::type
          , typename intrinsic_const_arg<A1>::type
        >
    {};

    template <class A0, class A1>
    typename F<A0,A1>::type
    operator()(A0 const& a0, A1 const& a1) const
    {
        return F<A0,A1>()(a0, a1);
    }
};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_FUNCTION2_DWA2005812_HPP
