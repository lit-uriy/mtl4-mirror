#ifndef STD_TUPLE_MANIP_HPP
#define STD_TUPLE_MANIP_HPP

#ifndef STD_TUPLE_IN_IO
#error "Must include <tupleio> for this library"
#endif

#include <iostream>
#include <string>

// Tuple I/O manipulators

namespace STD_TUPLE_NS {
  namespace detail {
    template <class Tag>
    struct string_ios_manip_helper {
      static int index() {
	static int index_ = std::ios_base::xalloc();
	return index_;
      }
    };

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
      }

      void set(const stringT& s) {
	stringT* p = (stringT*)(stream.pword(index));
	if (p) delete p;
	stream.pword(index) = (void*)(new stringT(s));
      }

      const stringT& get(const stringT& default_) const {
	if (stream.pword(index))
	  return *(stringT*)(stream.pword(index));
	else
	  return default_;
      }

      // FIXME: cleanup on program exit
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

}

#endif // STD_TUPLE_TUPLE0_HPP
