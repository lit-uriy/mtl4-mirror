#ifndef BOOST_TR1_TUPLE_MANIP_HPP
#define BOOST_TR1_TUPLE_MANIP_HPP

#ifndef BOOST_TR1_TUPLE_TUPLEIO_HPP
#error "Must include "tupleio.hpp" for this library"
#endif

#include <iostream>
#include <string>

// Tuple I/O manipulators
int my_index() {
  static int n = std::ios_base::xalloc();
  return n;
}


namespace boost {
  namespace tr1 {
    namespace tuple_detail {

      template <class Tag>
      struct string_ios_manip_helper {
	static int index() {
	  static int index_ = std::ios_base::xalloc();
	  return index_;
	}
      };

 
      struct any_string {
	virtual any_string* clone() = 0;
      };
      
      template <class Str> 
      class string_holder : public any_string {
	Str s;
      public:
	string_holder(const Str& s_) : s(s_) {}
	Str get() const { return s; }
	virtual string_holder* clone() { return new string_holder(s); }
      }     

      void tuple_manip_callback(std::ios_base::event ev,
				std::ios_base& b,
				int n) {
	any_string* p = (any_string*) b.pword(n);
	if (ev == std::ios_base::erase_event) {
	  delete p;
	  b.pword(n) = 0; 
	}
	else if (ev == std::ios_base::compyfmt_event && p != 0)
	  b.pword(n) = p->clone();
      }
	
      template <class Tag, class Stream>
      class string_ios_manip {
	// Based on article at 
	// http://www.cuj.com/experts/1902/austern.htm?topic=experts
	
	int index; Stream& stream;
	
	typedef std::basic_string<typename Stream::char_type,
				  typename Stream::traits_type> stringT;
	
      public:
	string_ios_manip(Stream& str_): stream(str_) {
	  index = string_ios_manip_helper<Tag>::index();
	  // FIXME: this might need to be in the setter instead?
	  int registered = stream.iword(index);
	  if (!registered) {
	    stream.iword(index) = 1;
	    stream.register_callback(tuple_manip_callback, index);
	  }
	}
	
	void set(const stringT& s) {
	  any_string* p = (any_string*)(stream.pword(index));
	  if (p) delete p;
	  stream.pword(index) = (void*)(new string_holder<stringT>(s));
	}
	
	const stringT& get(const stringT& default_) const {
	  if (stream.pword(index))
	    return ((string_holder<stringT>*)(stream.pword(index)))->get();
	  else
	    return default_;
	}
    };
  }

#define STD_TUPLE_DEFINE_MANIPULATOR(name) \
  namespace detail { \
    struct name##_tag; \
 \
    template <class CharT, class Traits = std::char_traits<CharT> > \
    struct name##_type { \
      typedef std::basic_string<CharT,Traits> stringT; \
      stringT data; \
      name##_type(const stringT& d): data(d) {} \
    }; \
 \
    template <class Stream, class CharT, class Traits> \
    Stream& \
    operator>> \
	(Stream& s, \
	 const name##_type<CharT,Traits>& m) { \
      string_ios_manip<name##_tag, Stream>(s). \
	set(m.data); \
      return s; \
    } \
 \
    template <class Stream, class CharT, class Traits> \
    Stream& \
    operator<< \
	(Stream& s, \
	 const name##_type<CharT,Traits>& m) { \
      string_ios_manip<name##_tag, Stream>(s). \
	set(m.data); \
      return s; \
    } \
  } \
  \
  template <class CharT, class Traits> \
  detail::name##_type<CharT,Traits> \
  name(const std::basic_string<CharT,Traits>& s) { \
    return detail::name##_type<CharT,Traits>(s); \
  } \
 \
  template <class CharT> \
  detail::name##_type<CharT> \
  name(const CharT c[]) { \
    return detail::name##_type<CharT>(std::basic_string<CharT>(c)); \
  } \
 \
  template <class CharT> \
  detail::name##_type<CharT> \
  name(CharT c) { \
    return detail::name##_type<CharT>(std::basic_string<CharT>(1,c)); \
  }

  STD_TUPLE_DEFINE_MANIPULATOR(tuple_open)
  STD_TUPLE_DEFINE_MANIPULATOR(tuple_close)
  STD_TUPLE_DEFINE_MANIPULATOR(tuple_delimiter)

#undef STD_TUPLE_DEFINE_MANIPULATOR

    } // namespace tuple_detail
  } // namespace tr1
} // namespace boost

#endif // BOOST_TR1_TUPLE_MANIP_HPP
