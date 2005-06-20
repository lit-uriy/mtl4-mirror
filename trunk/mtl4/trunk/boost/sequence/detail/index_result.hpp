// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_INDEX_RESULT_DWA2005617_HPP
# define BOOST_SEQUENCE_DETAIL_INDEX_RESULT_DWA2005617_HPP

# include <boost/type_traits/remove_reference.hpp>
# include <boost/sequence/detail/transfer_cv.hpp>
# include <boost/mpl/if.hpp>

namespace boost { namespace sequence { namespace detail { 

template <class A, class N>
struct index_result
{
    typedef typename remove_reference<A>::type a;
    typedef typename a::reference reference;
    typedef typename remove_reference<reference>::type value;
    
    typedef typename mpl::if_<
        is_reference<reference>
      , typename transfer_cv<a,value>::type&
      , reference
    >::type type;
};

template <class T, std::size_t N, class M>
struct index_result<T[N],M>
{
    typedef T& type;
};

template <class T, std::size_t N, class M>
struct index_result<T(&)[N],M>
{
    typedef T& type;
};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_INDEX_RESULT_DWA2005617_HPP
