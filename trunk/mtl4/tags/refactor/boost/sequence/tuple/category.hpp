// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_TUPLE_SEQUENCE_CATEGORY_DWA200559_HPP
# define BOOST_TUPLE_SEQUENCE_CATEGORY_DWA200559_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence { 

struct tuple_tag
  : o1_size_tag
{};

template <class T, std::size_t N>
struct category<T[N]>
{
    typedef tuple_tag type;
};

template <class T, std::size_t N>
struct category<T const[N]>
{
    typedef tuple_tag type;
};

template <class T, std::size_t N>
struct category<boost::array<T,N> >
{
    typedef tuple_tag type;
};

}} // namespace boost::sequence

#endif // BOOST_TUPLE_SEQUENCE_CATEGORY_DWA200559_HPP
