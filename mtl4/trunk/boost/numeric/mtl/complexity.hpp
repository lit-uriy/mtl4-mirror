// $COPYRIGHT$

#ifndef MTL_COMPLEXITY_INCLUDE
#define MTL_COMPLEXITY_INCLUDE

#include <boost/mpl/int.hpp>
#include <boost/mpl/max.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/less.hpp>

// This file contains types to characterize time complexities of operations or traversals
// Different traversals over given collections have different run time
// Algorithms implementable with different traversals can use these complexities to dispatch

// Complexities in a order, which of course will not be changed
// The underlying MPL definitions might be modified to add finer grained distinctions

// Summation and multiplication of complexities is planned

namespace mtl { namespace complexity {

  // namespace mpl = boost::mpl;

// Infinite time complexity, which usually means that the operation or traversal is not available
struct infinite : boost::mpl::int_<1000> {};

struct polynomial : boost::mpl::int_<200> {};

struct quadratic : boost::mpl::int_<100> {};

struct n_log_n : boost::mpl::int_<20> {};

struct linear : boost::mpl::int_<11> {};

// Product of linear and cached
struct linear_cached : boost::mpl::int_<10> {};

struct log : boost::mpl::int_<5> {};

struct constant : boost::mpl::int_<2> {};

// Special type for traversals to distinguish between strided or random memory access with 'constant' 
// (but slow) memory access and consecutive memory access with a good change that only one element
// per cache line must be load from memory
struct cached : boost::mpl::int_<1> {};

// template <typename X, typename Y> 
// struct plus : boost::mpl::max<X, Y> {}

// Adding complexities of two operations is the maximal complexity of both operations
template <typename X, typename Y> 
struct plus : boost::mpl::if_< boost::mpl::less<X, Y>, Y, X> {};


namespace detail
{
    template <typename X, typename Y> struct times {};

    template<typename Y> struct times<cached, Y> 
    {
	typedef Y type; 
    };

    template<typename Y> struct times<constant, Y> 
    {
	typedef Y type; 
    };   

} // namespace detail

// Multiplication needs to be defined explicitly
// At least is symmetric, so we only consider X <= Y
template <typename X, typename Y> 
struct times
    : boost::mpl::if_< 
            boost::mpl::less<X, Y>
          , detail::times<X, Y>
          , detail::times<Y, X>
          >
{};

}} // namespace mtl

#endif // MTL_COMPLEXITY_INCLUDE
