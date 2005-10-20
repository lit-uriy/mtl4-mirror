// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_CURSOR_DWA2005330_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_CURSOR_DWA2005330_HPP

# include <boost/sequence/fixed_size/cursor_fwd.hpp>
# include <boost/sequence/detail/is_mpl_integral_constant.hpp>
# include <boost/sequence/detail/typeof_add.hpp>
# include <boost/sequence/detail/typeof_subtract.hpp>
# include <boost/sequence/intrinsic/operations.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/type_traits/is_class.hpp>
# include <boost/type_traits/is_integral.hpp>
# include <boost/mpl/integral_c.hpp>
# include <boost/typeof/typeof.hpp>
# include <cstddef>

namespace boost {
namespace sequence {
namespace fixed_size { 

  namespace cursor_ // namespace for ADL protection.
  {
    // A cursor that stores a compile-time index.
    template <std::size_t N>
    struct cursor
    {
        static std::size_t const value = N;
        operator std::size_t() const { return N; }
        cursor operator*() const { return *this; }
        typedef cursor type;
    };

    // The difference of two cursors is an integral constant
    template <std::size_t N1, std::size_t N2>
    inline mpl::integral_c<std::ptrdiff_t,(N1-N2)>
    operator-(cursor<N1>, cursor<N2>)
    {
        return mpl::int_<(N1-N2)>();
    }

    // Relational operators returning mpl::bool_ constants.
  # define BOOST_SEQUENCE_fixed_size_cursor_relational(op) \
    template <std::size_t N1, std::size_t N2>              \
    inline mpl::bool_<(N1 op N2)>                          \
    operator op(cursor<N1>,cursor<N2>)                     \
    {                                                      \
        return mpl::bool_<(N1 op N2)>();                   \
    }

    BOOST_SEQUENCE_fixed_size_cursor_relational(==)
    BOOST_SEQUENCE_fixed_size_cursor_relational(!=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(<)
    BOOST_SEQUENCE_fixed_size_cursor_relational(<=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(>=)
    BOOST_SEQUENCE_fixed_size_cursor_relational(>)

  # undef BOOST_SEQUENCE_fixed_size_cursor_relational

    //
    // Random access interactions with integral constants -- these
    // operations preserve the integrity of the cursor's position as a
    // compile-time constant.
    //

    // helper metafunctions
    template <std::size_t N, class T>
    struct add
    {
        typedef cursor<(N + T::value)> type;
    };

    template <std::size_t N, class T>
    struct subtract
    {
        typedef cursor<(N - T::value)> type;
    };


    template <std::size_t N, class T>
    typename lazy_enable_if<
        typename detail::is_mpl_integral_constant<T>::type
      , cursor_::add<N,T>
    >::type
    inline operator+(cursor<N>, T)
    {
        return cursor<(N+T::value)>();
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        typename detail::is_mpl_integral_constant<T>::type
      , cursor_::add<N,T>
    >::type
    inline operator+(T,cursor<N>)
    {
        return cursor<(N+T::value)>();
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        typename detail::is_mpl_integral_constant<T>::type
      , cursor_::subtract<N,T>
    >::type
    inline operator-(cursor<N>, T)
    {
        return cursor<(N-T::value)>();
    }

    //
    // Random access interactions with runtime integers.
    //

    template <std::size_t N, class T>
    typename lazy_enable_if<
        boost::is_integral<T>
      , detail::typeof_add<T,std::size_t>
    >::type
    inline operator+(cursor<N>, T x)
    {
        return x + N;
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        boost::is_integral<T>
      , detail::typeof_add<T,std::size_t>
    >::type
    inline operator+(T x,cursor<N>)
    {
        return x + N;
    }

    template <std::size_t N, class T>
    typename lazy_enable_if<
        boost::is_integral<T>
      , detail::typeof_subtract<std::size_t,T>
    >::type
    inline operator-(cursor<N>, T x)
    {
        return N - x;
    }
  } // namespace cursor_

} // namespace fixed_size

namespace intrinsic
{
  // Increment and decrement
  template <std::size_t N>
  struct next<fixed_size::cursor<N> >
  {
      typedef fixed_size::cursor<(N+1)> type;
      type operator()(fixed_size::cursor<N> const&) const
      {
          return type();
      }
  };

  template <std::size_t N>
  struct prev<fixed_size::cursor<N> >
  {
      typedef fixed_size::cursor<(N-1)> type;
      type operator()(fixed_size::cursor<N> const&) const
      {
          return type();
      }
  };

  template <std::size_t N, class D>
  struct advance<fixed_size::cursor<N>, D>
  {
      typedef fixed_size::cursor<(N+D::value)> type;
      type operator()(fixed_size::cursor<N> const&, D const&) const
      {
          return type();
      }
  };

  template <std::size_t N0, std::size_t N1>
  struct distance<fixed_size::cursor<N0>, fixed_size::cursor<N1> >
  {
      typedef mpl::integral_c<std::ptrdiff_t,(N1-N0)> type;
      
      type operator()(fixed_size::cursor<N0> const&, fixed_size::cursor<N1> const&) const
      {
          return type();
      }
  };
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_CURSOR_DWA2005330_HPP
