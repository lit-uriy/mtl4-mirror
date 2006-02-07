// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef TRANSFER_CV_DWA200541_HPP
# define TRANSFER_CV_DWA200541_HPP

namespace boost { namespace sequence { 
namespace detail { 

// Applies any cv-qualification on Source to Destination.
template <class Source, class Destination>
struct transfer_cv
{
    typedef Destination type;
};

template <class Source, class Destination>
struct transfer_cv<Source const, Destination>
{
    typedef Destination const type;
};

template <class Source, class Destination>
struct transfer_cv<Source volatile, Destination>
{
    typedef Destination volatile type;
};

template <class Source, class Destination>
struct transfer_cv<Source const volatile, Destination>
{
    typedef Destination const volatile type;
};

}}} // namespace boost::sequence::detail

#endif // TRANSFER_CV_DWA200541_HPP
