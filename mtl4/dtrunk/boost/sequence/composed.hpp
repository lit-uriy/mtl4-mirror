// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_COMPOSED_DWA2006513_HPP
# define BOOST_SEQUENCE_COMPOSED_DWA2006513_HPP

# include <boost/property_map/identity.hpp>
# include <boost/compressed_pair.hpp>
# include <boost/sequence/intrinsics.hpp>

namespace boost { namespace sequence { 

template <class Cursor, class Mapping = property_map::identity>
struct composed
{
    typedef Cursor cursor;
    typedef Mapping mapping;
    
    composed(Cursor start, Cursor finish, Mapping mapping)
      : start(start)
      , finish_and_mapping(finish, mapping)
    {}

    Cursor begin() const { return this->start; }
    Cursor end() const { return this->finish_and_mapping.first(); }
    mapping elements() const { return this->finish_and_mapping.second(); }

 private:
    Cursor start;
    compressed_pair<Cursor,Mapping> finish_and_mapping;
};

namespace impl
{
  struct composed_tag {};
  
  template <class C, class M>
  struct tag<composed<C,M> >
  {
      typedef composed_tag type;
  };

  template <class S>
  struct intrinsics<S, composed_tag>
  {
      struct begin
      {
          typedef typename S::cursor result_type;
          result_type operator()(S& s) const
          {
              return s.begin();
          }
      };
      
      struct end
      {
          typedef typename S::cursor result_type;
          result_type operator()(S& s) const
          {
              return s.end();
          }
      };

      struct elements
      {
          typedef typename S::mapping result_type;
          result_type operator()(S& s) const
          {
              return s.elements();
          }
      };
  };
  
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_COMPOSED_DWA2006513_HPP
