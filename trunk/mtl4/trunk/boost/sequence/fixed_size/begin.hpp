// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_BEGIN_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_BEGIN_DWA200555_HPP

# include <boost/sequence/intrinsic/begin_fwd.hpp>
# include <boost/sequence/fixed_size/cursor.hpp>
# include <boost/array.hpp>

# include <cstddef>

#error obsolete 
namespace boost { namespace sequence { namespace intrinsic {

template <class T, std::size_t N>
struct begin<T[N]>
{
    typedef fixed_size::cursor<0> type;
    type operator()(T(&)[N]) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct begin<array<T,N> >
{
    typedef fixed_size::cursor<0> type;
    type operator()(array<T,N>&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct begin<array<T,N> const>
{
    typedef fixed_size::cursor<0> type;
    type operator()(array<T,N> const&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct begin<array<T,N> volatile>
{
    typedef fixed_size::cursor<0> type;
    type operator()(array<T,N> volatile&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct begin<array<T,N> const volatile>
{
    typedef fixed_size::cursor<0> type;
    type operator()(array<T,N> const volatile&) const
    {
        return type();
    }
};
                                       
}}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_BEGIN_DWA200555_HPP
