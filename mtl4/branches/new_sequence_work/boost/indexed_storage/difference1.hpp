// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_DIFFERENCE1_DWA200659_HPP
# define BOOST_INDEXED_STORAGE_DIFFERENCE1_DWA200659_HPP

# include <boost/iterator/iterator_traits.hpp>

namespace boost { namespace indexed_storage { 

template <class Base>
struct difference1
{
    typedef typename iterator_difference<Base>::type result_type;

    explicit difference1(Base x)
      : base(x)
    {}

    result_type operator()(Base const& y) const
    {
        return y - base;
    }
        
 private:
    Base base;
};

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_DIFFERENCE1_DWA200659_HPP
