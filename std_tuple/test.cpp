#include "tuple"
#include <string>
#include <iostream>

template <class O, int Start, int Len, class T>
struct print_tuple_type {
  static void run(O& o, const T& x) {
    o << std_tuple::get<Start>(x);
    if (Start+1 != Len) o << " ";
    print_tuple_type<O,Start+1, Len, T>::run(o,x);
  }
};

template <class O, int Len, class T>
struct print_tuple_type<O,Len,Len,T> {
  static void run(O& o, const T&) {
    o << "]";
  }
};

template <class O, class T>
void print_tuple(O& o, const T& x) {
  std::cout << "[";
  print_tuple_type<O, 0, std_tuple::tuple_size<T>::value, T>::run(o,x);
}

template <class O, class T>
typename std_tuple::detail::enable_if<
  std_tuple::is_tuple<T>::value,
  O&>::type
operator << (O& o, const T& x) {
  print_tuple(o,x);
  return o;
}

struct dummy_type {
  operator double() const {return 0;}
  operator std_tuple::tuple<double>() const {return std_tuple::tuple<double>(1);}
  // operator std_tuple::tuple<std_tuple::tuple<double> >() const {return 2;}
};

int f(int a, int b) {
  std::cout << "a=" << a << " b='" << b << "'" << std::endl;
  return a+b;
}

struct fcall {
  int a,b;
  fcall(int a_, int b_): a(a_), b(b_) {}
};

extern "C" int tuple_call_2(int (*)(...), void*, int);

template <class Ret, class F, class Tuple>
Ret tuple_call(F f, const Tuple& tuple) {
  return tuple_call_2((int (*)(...))&f, (void*)&tuple, sizeof(Tuple));
}

int main(int, char**) {
  std_tuple::tuple<int, const int, std::string> a(1, 'a', "Hi");
  std::cout << a << std::endl;
  char c = 'c';
  std::cout << std_tuple::make_tuple(1,5,make_tuple(make_tuple(a)),(c)) << std::endl;
  char b = 'a';
  std::cout << b << std::endl;
  std_tuple::tie(b, std_tuple::ignore) = std_tuple::make_tuple(c, 100);
  std::cout << b << std::endl;
  std_tuple::tuple<dummy_type> x;
  dummy_type y;
  std::cout << std_tuple::tuple<double>(x) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<double> >(x) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<std_tuple::tuple<double> > >(x) << std::endl;
  // std::cout << std_tuple::tuple<std_tuple::tuple<double> >(std_tuple::make_tuple(x)) << std::endl;
  // std::cout << std_tuple::tuple<std_tuple::tuple<double> >(std_tuple::make_tuple(std_tuple::make_tuple(x))) << std::endl;
  std::cout << std_tuple::tuple<double>(y) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<double> >(y) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<std_tuple::tuple<double> > >(y) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<double> >(std_tuple::make_tuple(y)) << std::endl;
  std::cout << std_tuple::tuple<std_tuple::tuple<double> >(std_tuple::make_tuple(std_tuple::make_tuple(y))) << std::endl;

  std::cout << std_tuple::tuple<char&, char>(c,c) << std::endl;
  std::cout << std_tuple::make_tuple(std_tuple::ref(c), c) << std::endl;
  std::cout << std_tuple::make_tuple(std_tuple::cref(c), c) << std::endl;
  std::cout << std_tuple::make_tuple(std_tuple::ref(a), c) << std::endl;
  // std::cout << tuple_call<int>(&f, fcall(1,2)) << std::endl;
  std::cout << tuple_call<int>(&f, std_tuple::make_tuple(1, 2)) << std::endl;
  return 0;
}
