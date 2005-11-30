// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSIC_MINUS_DWA20051128_HPP
# define BOOST_SEQUENCE_INTRINSIC_MINUS_DWA20051128_HPP

# include <boost/typeof/typeof.hpp>
# include <boost/sequence/detail/is_mpl_integral_constant.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/mpl/and.hpp>
# include <boost/mpl/minus.hpp>

namespace boost { namespace sequence { namespace intrinsic { 
  
template <class N1, class N2, class enable = void>
struct minus_impl
{
    static N1 const n1;
    static N2 const n2;
    
    typedef BOOST_TYPEOF_TPL(n1 - n2) type;
    
    type operator()(N1 const& x, N2 const& y) const
    {
        return x - y;
    }
};

template <class, class> class undef;

template <class N1, class N2>
struct minus_impl<
    N1
  , N2
  , typename enable_if<
        mpl::and_<
            detail::is_mpl_integral_constant<N1>
          , detail::is_mpl_integral_constant<N2>
        >
    >::type
>
{
    typedef typename mpl::minus<N1,N2>::type type;
      
    type operator()(N1 const& c1, N2 const& c2) const
    {
        return type();
    }
};

template <class N1, class N2>
struct minus: minus_impl<N1,N2> {};

template <class N1, class N2>
struct minus<N1,N2 const> : minus<N1,N2> {};

template <class N1, class N2>
struct minus<N1 const,N2> : minus<N1,N2> {};

template <class N1, class N2>
struct minus<N1 const,N2 const> : minus<N1,N2> {};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_INTRINSIC_MINUS_DWA20051128_HPP
