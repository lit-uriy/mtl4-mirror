// -*- mode: c++; -*-

#include <istream>
#include <ostream>

#include "boost/tuple/tuple.hpp"

namespace std {

  namespace tuple_detail {

    class format_info {
    public:   

      enum manipulator_type { open, close, delimiter };
      BOOST_STATIC_CONSTANT(int, number_of_manipulators = delimiter + 1);
    private:
      
      static int get_stream_index (int m)
      {
	static const int stream_index[number_of_manipulators]
	  = { std::ios::xalloc(), std::ios::xalloc(), std::ios::xalloc() };
	
	return stream_index[m];
      }
      
      format_info(const format_info&);
      format_info();   
      
      
    public:
      
      template<class CharType, class CharTrait>
      static CharType get_manipulator(std::basic_ios<CharType, CharTrait>& i, 
				      manipulator_type m) {
	// The manipulators are stored as long.
	// A valid instanitation of basic_stream allows CharType to be any POD,
	// hence, the static_cast may fail (it fails if long is not convertible 
	// to CharType
	CharType c = static_cast<CharType>(i.iword(get_stream_index(m)) ); 
	// parentheses and space are the default manipulators
	if (!c) {
	  switch(m) {
	  case open :  c = i.widen('('); break;
	  case close : c = i.widen(')'); break;
	  case delimiter : c = i.widen(' '); break;
	  }
	}
	return c;
      }
      
      
      template<class CharType, class CharTrait>
   template<class CharTrait>
  void set(std::basic_ios<CharType, CharTrait> &io) const {
     tuple_detail::format_info::set_manipulator(io, mt, f_c);
  }
};


template<class CharType, class CharTrait>
inline std::basic_ostream<CharType, CharTrait>&
operator<<(std::basic_ostream<CharType, CharTrait>& o, const tuple_manipulator<CharType>& m) {
  m.set(o);
  return o;
}

template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>&
operator>>(std::basic_istream<CharType, CharTrait>& i, const tuple_manipulator<CharType>& m) {
  m.set(i);
  return i;
}

   
template<class CharType>
inline tuple_manipulator<CharType> set_open(const CharType c) {
   return tuple_manipulator<CharType>(tuple_detail::format_info::open, c);
}

template<class CharType>
inline tuple_manipulator<CharType> set_close(const CharType c) {
   return tuple_manipulator<CharType>(tuple_detail::format_info::close, c);
}

template<class CharType>
inline tuple_manipulator<CharType> set_delimiter(const CharType c) {
   return tuple_manipulator<CharType>(tuple_detail::format_info::delimiter, c);
}



   
   
// -------------------------------------------------------------
// printing tuples to ostream in format (a b c)
// parentheses and space are defaults, but can be overriden with manipulators
// set_open, set_close and set_delimiter
   
namespace tuple_detail {

// Note: The order of the print functions is critical 
// to let a conforming compiler  find and select the correct one.


template<class CharType, class CharTrait, class T1>
inline std::basic_ostream<CharType, CharTrait>& 
print(std::basic_ostream<CharType, CharTrait>& o, const cons<T1, null_type>& t) {
  return o << t.head;
}

 
template<class CharType, class CharTrait>
inline std::basic_ostream<CharType, CharTrait>& 
print(std::basic_ostream<CharType, CharTrait>& o, const null_type&) { 
  return o; 
}

template<class CharType, class CharTrait, class T1, class T2>
inline std::basic_ostream<CharType, CharTrait>& 
print(std::basic_ostream<CharType, CharTrait>& o, const cons<T1, T2>& t) {
  
  const CharType d = format_info::get_manipulator(o, format_info::delimiter);
   
  o << t.head;

  o << d;

  return print(o, t.tail);
}


} // namespace tuple_detail

#if defined (BOOST_NO_TEMPLATED_STREAMS)
template<class T1, class T2>
inline std::ostream& operator<<(std::ostream& o, const cons<T1, T2>& t) {
  if (!o.good() ) return o;
 
  const char l = 
    tuple_detail::format_info::get_manipulator(o, tuple_detail::format_info::open);
  const char r = 
    tuple_detail::format_info::get_manipulator(o, tuple_detail::format_info::close);
   
  o << l;
  
  tuple_detail::print(o, t);  

  o << r;

  return o;
}

#else

template<class CharType, class CharTrait, class T1, class T2>
inline std::basic_ostream<CharType, CharTrait>& 
operator<<(std::basic_ostream<CharType, CharTrait>& o, 
           const cons<T1, T2>& t) {
  if (!o.good() ) return o;
 
  const CharType l = 
    tuple_detail::format_info::get_manipulator(o, tuple_detail::format_info::open);
  const CharType r = 
    tuple_detail::format_info::get_manipulator(o, tuple_detail::format_info::close);
   
  o << l;   

  tuple_detail::print(o, t);  

  o << r;

  return o;
}
#endif // BOOST_NO_TEMPLATED_STREAMS

   
// -------------------------------------------------------------
// input stream operators

namespace tuple_detail {

#if defined (BOOST_NO_TEMPLATED_STREAMS)

inline std::istream& 
extract_and_check_delimiter(
  std::istream& is, format_info::manipulator_type del)
{
  const char d = format_info::get_manipulator(is, del);

  const bool is_delimiter = (!isspace(d) );      

  char c;
  if (is_delimiter) { 
    is >> c; 
    if (c!=d) {
      is.setstate(std::ios::failbit);
    } 
  }
  return is;
}


// Note: The order of the read functions is critical to let a 
// (conforming?) compiler find and select the correct one.

#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
template<class T1>
inline  std::istream & 
read (std::istream &is, cons<T1, null_type>& t1) {

  if (!is.good()) return is;   
   
  return is >> t1.head ;
}
#else
inline std::istream& read(std::istream& i, const null_type&) { return i; }
#endif // !BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
   
template<class T1, class T2>
inline std::istream& 
read(std::istream &is, cons<T1, T2>& t1) {

  if (!is.good()) return is;
   
  is >> t1.head;

#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
  if (tuples::length<T2>::value == 0)
    return is;
#endif  // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

  extract_and_check_delimiter(is, format_info::delimiter);

  return read(is, t1.tail);
}

} // end namespace tuple_detail

inline std::istream& 
operator>>(std::istream &is, null_type&) {

  if (!is.good() ) return is;

  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::open);
  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::close);

  return is;
}


template<class T1, class T2>
inline std::istream& 
operator>>(std::istream& is, cons<T1, T2>& t1) {

  if (!is.good() ) return is;

  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::open);
                      
  tuple_detail::read(is, t1);
   
  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::close);

  return is;
}



#else

template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>& 
extract_and_check_delimiter(
  std::basic_istream<CharType, CharTrait> &is, format_info::manipulator_type del)
{
  const CharType d = format_info::get_manipulator(is, del);

  const bool is_delimiter = (!isspace(d) );      

  CharType c;
  if (is_delimiter) { 
    is >> c;
    if (c!=d) { 
      is.setstate(std::ios::failbit);
    }
  }
  return is;
}

   
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
template<class CharType, class CharTrait, class T1>
inline  std::basic_istream<CharType, CharTrait> & 
read (std::basic_istream<CharType, CharTrait> &is, cons<T1, null_type>& t1) {

  if (!is.good()) return is;   
   
  return is >> t1.head; 
}
#else
template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>& 
read(std::basic_istream<CharType, CharTrait>& i, const null_type&) { return i; }

#endif // !BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template<class CharType, class CharTrait, class T1, class T2>
inline std::basic_istream<CharType, CharTrait>& 
read(std::basic_istream<CharType, CharTrait> &is, cons<T1, T2>& t1) {

  if (!is.good()) return is;
   
  is >> t1.head;

#if defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
  if (tuples::length<T2>::value == 0)
    return is;
#endif  // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

  extract_and_check_delimiter(is, format_info::delimiter);

  return read(is, t1.tail);
}

} // end namespace tuple_detail


template<class CharType, class CharTrait>
inline std::basic_istream<CharType, CharTrait>& 
operator>>(std::basic_istream<CharType, CharTrait> &is, null_type&) {

  if (!is.good() ) return is;

  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::open);
  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::close);

  return is;
}

template<class CharType, class CharTrait, class T1, class T2>
inline std::basic_istream<CharType, CharTrait>& 
operator>>(std::basic_istream<CharType, CharTrait>& is, cons<T1, T2>& t1) {

  if (!is.good() ) return is;

  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::open);
                      
  tuple_detail::read(is, t1);
   
  tuple_detail::extract_and_check_delimiter(is, tuple_detail::format_info::close);

  return is;
}

