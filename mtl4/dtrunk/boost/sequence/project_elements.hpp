// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_PROJECT_ELEMENTS_DWA2006516_HPP
# define BOOST_SEQUENCE_PROJECT_ELEMENTS_DWA2006516_HPP

# include <boost/sequence/concepts.hpp>
# include <boost/property_map/compose.hpp>
# include <boost/utility/result_of.hpp>

namespace boost { namespace sequence { 

// A function object that takes an instance of S and yields an
// accessor that is a projection of the instance's accessor.
template <class S, class Projection>
struct project_elements
{
    BOOST_CONCEPT_ASSERT((concepts::Sequence<S>));

    typedef typename result_of<
        property_map::op::compose(
            Projection
          , typename concepts::Sequence<S>::elements
        )
    >::type result_type;
      
    BOOST_CONCEPT_ASSERT((
         concepts::ReadablePropertyMap<
             result_type
           , typename concepts::Sequence<S>::cursor
         >
    ));
    
    result_type operator()(S& s) const
    {
        return result_type(Projection(), elements(s));
    }
};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_PROJECT_ELEMENTS_DWA2006516_HPP
