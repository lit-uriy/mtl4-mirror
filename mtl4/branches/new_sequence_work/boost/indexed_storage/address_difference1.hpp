// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_ADDRESS_DIFFERENCE1_DWA200659_HPP
# define BOOST_INDEXED_STORAGE_ADDRESS_DIFFERENCE1_DWA200659_HPP

# include <boost/implicit_cast.hpp>
# include <cstdlib>

namespace boost { namespace indexed_storage { 

template <class Base>
struct address_difference1
{
    typedef std::ptrdiff_t result_type;
    
    explicit address_difference1(Base x)
      : base(x)
    {}

    result_type operator()(
        typename remove_pointer<Base>::type const& y) const
    {
        return &y - base;
    }
        
 private:
    Base base;
};

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_ADDRESS_DIFFERENCE1_DWA200659_HPP
