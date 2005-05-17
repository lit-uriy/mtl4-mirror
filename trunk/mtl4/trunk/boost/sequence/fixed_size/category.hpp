// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
# define BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/fixed_size/is_fixed_size.hpp>
# include <boost/sequence/algorithm/fixed_size/category.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence { 

template <class Sequence>
struct category<
    Sequence
  , typename enable_if<fixed_size::is_fixed_size<Sequence> >::type
>
{
    typedef algorithm::fixed_size::category type;
};

}} // namespace boost::sequence

#endif // BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
