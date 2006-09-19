// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FUSION_COUNTING_SEQUENCE_DWA2006919_HPP
# define BOOST_FUSION_COUNTING_SEQUENCE_DWA2006919_HPP

# include <boost/fusion/support/iterator_base.hpp>
# include <boost/fusion/support/category_of.hpp>
# include <boost/detail/transfer_cv.hpp>
# include <boost/type_traits/add_reference.hpp>

namespace boost { namespace fusion { 

template <class Sequence>
struct counting_sequence
{
    Sequence
};

}} // namespace boost::fusion

#endif // BOOST_FUSION_COUNTING_SEQUENCE_DWA2006919_HPP
