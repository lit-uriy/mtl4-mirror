// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_TRAITS_REFERENCE_DWA2006926_HPP
# define BOOST_SEQUENCE_TRAITS_REFERENCE_DWA2006926_HPP

# include <boost/sequence/traits/is_homogeneous.hpp>

namespace boost { namespace sequence { namespace traits { 

// A metafunction that returns the result type of reading a
// homogeneous sequence of type S.
template <class S>
struct reference
{
    typedef typename Sequence<S>::begin_cursor cursor;
    typedef typename 
    typedef typename result_of<
        deref
    
};
    deref

}}} // namespace boost::sequence::traits

#endif // BOOST_SEQUENCE_TRAITS_REFERENCE_DWA2006926_HPP
