// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP

# include <boost/sequence/intrinsic/elements_fwd.hpp>
# include <boost/sequence/index_property_map.hpp>
# include <boost/array.hpp>

# include <cstddef>

#error obsolete 
namespace boost { namespace sequence { namespace intrinsic {

template <class T, std::size_t N>
struct elements<T[N]>
{
    typedef index_property_map<T(&)[N]> type;
    type operator()(T(&)[N]) const
    {
        return type();
    }
};
                                        
template <class T, std::size_t N>
struct elements<array<T,N> >
{
    typedef index_property_map<array<T,N>&> type;
    type operator()(array<T,N>&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct elements<array<T,N> const>
{
    typedef index_property_map<array<T,N> const&> type;
    type operator()(array<T,N> const&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct elements<array<T,N> volatile>
{
    typedef index_property_map<array<T,N> volatile&> type;
    type operator()(array<T,N> volatile&) const
    {
        return type();
    }
};
                                       
template <class T, std::size_t N>
struct elements<array<T,N> const volatile>
{
    typedef index_property_map<array<T,N> const volatile&> type;
    type operator()(array<T,N> const volatile&) const
    {
        return type();
    }
};
                                       
}}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
