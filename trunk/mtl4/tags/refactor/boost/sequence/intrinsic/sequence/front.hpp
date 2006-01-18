// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FRONT_DWA2005511_HPP
# define BOOST_SEQUENCE_FRONT_DWA2005511_HPP

# include <boost/utility/result_of.hpp>
# include <boost/sequence/intrinsic/sequence/elements.hpp>
# include <boost/sequence/intrinsic/sequence/begin.hpp>
# include <boost/sequence/accessor.hpp>
# include <boost/sequence/dereferenced.hpp>
# include <boost/sequence/intrinsic/sequence/begin.hpp>

namespace boost { namespace sequence { 

template <class Sequence>
typename result_of<
    typename accessor<Sequence>::type(
        typename dereferenced<
            typename intrinsic::begin<Sequence>::type
        >::type
    )
>::type
front(Sequence const& s)
{
    return sequence::elements(s)( *sequence::begin(s) )
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FRONT_DWA2005511_HPP
