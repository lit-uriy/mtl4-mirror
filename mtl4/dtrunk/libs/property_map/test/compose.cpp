// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/property_map/compose.hpp>
#include <boost/property_map/dereference.hpp>
#include <boost/property_map/identity.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/detail/transfer_cv.hpp>
#include <boost/detail/project1st.hpp>
#include <boost/detail/project2nd.hpp>
#include <functional>

using namespace boost;

typedef counting_iterator<int*> iter;
typedef counting_iterator<int const*> citer;

result_of<
    property_map::op::compose(
        std::binder2nd<std::multiplies<int> >
      , property_map::dereference
    )
>::type deref3(
    // second argument is stateless
    std::bind2nd(std::multiplies<int>(),3)
);

int values[] = { 0, 1, 2, 3, 4 };

struct pair
{
    typedef int first_type;
    typedef char second_type;
    int first;
    char second;
};

pair pairs[] = { {1,'b'}, {2,'c'}, {3,'d'}, {4,'e'}, {5,'f'}};

typedef result_of<
    property_map::op::compose(
        boost::detail::project1st
      , property_map::identity
    )
>::type first_t;

first_t first;

result_of<
    property_map::op::compose(
        boost::detail::project2nd
      , property_map::identity
    )
>::type second;

int main()
{
    iter i(values);
    citer ci(values);

    for (int n = 0; n < 5; n++)
    {
        BOOST_TEST( deref3( *(i+n) ) == 3*n );
        BOOST_TEST( deref3( *(ci+n) ) == 3*n );

        if (n > 2)
        {
            // Check that the global compose function works and
            // doesn't change anything.
            deref3 = property_map::compose(
                std::bind2nd(std::multiplies<int>(),3)
              , property_map::dereference()
            );
        }

        BOOST_TEST( first( *(pairs + n) ) == n+1 );
        BOOST_TEST( second( *(pairs + n) ) == char('a'+n+1) );
        first( *(pairs + n), n*2 );
        BOOST_TEST( first( *(pairs + n) ) == n*2 );
    }
}

