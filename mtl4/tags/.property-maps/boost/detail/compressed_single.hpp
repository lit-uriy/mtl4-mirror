// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_DETAIL_COMPRESSED_SINGLE_DWA2006519_HPP
# define BOOST_DETAIL_COMPRESSED_SINGLE_DWA2006519_HPP

# include <boost/type_traits/is_class.hpp>
# include <boost/type_traits/is_convertible.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace detail { 

template <class T, bool empty>
struct compressed_single_base;

template <class T>
struct compressed_single_base<T,true>
  : private T
{
    compressed_single_base() {}
    
    compressed_single_base(T const& x)
      : T(x)
    {}
    
    compressed_single_base(T& x) // e.g., for auto_ptr
      : T(x)
    {}
    
    T& first() { return *this; }
    T const& first() const { return *this; }
};

template <class T>
struct compressed_single_base<T const,true>
  : private T
{
    compressed_single_base() {}
    
    compressed_single_base(T const& x)
      : T(x)
    {}
    
    compressed_single_base(T& x) // e.g., for auto_ptr
      : T(x)
    {}
    
    T const& first() const { return *this; }
};

template <class T>
struct compressed_single_base<T,false>
{
    compressed_single_base() {}
    
    compressed_single_base(T const& x)
      : x(x)
    {}
    
    compressed_single_base(T& x) // e.g., for auto_ptr
      : x(x)
    {}
    
    T& first() { return x; }
    T const& first() const { return x; }
    
 private:
    T x;
};

template <class T>
struct compressed_single_base<T const,false>
{
    compressed_single_base() {}
    
    compressed_single_base(T const& x)
      : x(x)
    {}
    
    compressed_single_base(T& x) // e.g., for auto_ptr
      : x(x)
    {}
    
    T const& first() const { return x; }
    
 private:
    T const x;
};

template <class T>
struct compressed_single_base<T&,false>
{
    compressed_single_base() {}
    
    compressed_single_base(T& x)
      : x(x)
    {}
    
    T& first() const { return x; }
    
 private:
    T& x;
};


template <class T>
struct compressed_single
  : compressed_single_base<T, is_class<T>::value>
{
    typedef T first_type;
    typedef void second_type;   // for uniformity w/compressed_tuple
    
    typedef compressed_single_base<T,is_class<T>::value> base;
    
    template <class U>
    compressed_single(
        compressed_single<U> const& u
      , typename enable_if<is_convertible<T,U> >::type* = 0
    )
      : base(u.first)
    {}

    explicit compressed_single(T const& x)
      : base(x)
    {}
};

}} // namespace boost::detail

#endif // BOOST_DETAIL_COMPRESSED_SINGLE_DWA2006519_HPP
