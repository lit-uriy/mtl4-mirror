#include <iostream>

#include "boost/tuple/tuple.hpp"

class A {
public:
  virtual ~A() {};
  const int i;  

  A() /*: i()*/ {}
};

class B {
  B(const B& b);
public:
  B() {}
};

namespace AA {
  template <class T> class X;
}
namespace AA {
  template <> class X<int> {};
}
 
  using namespace boost::tuples;
  typedef cons<int, cons<int, null_type> > tt;

tt foo() { return tt(); }

int main() {

  boost::tuple<B> b;
#if 0  
boost::tuple<const int> i;
  std::cout << i.head << std::endl;
  A a;
  std::cout << a.i << std::endl;

  boost::tuple<int, int> bb(std::make_pair(1, 3.14));  
#endif
 

  typedef cons<int, cons<int, null_type> > tt;
 
  tt t;
  tt k = t;
  tt kk(foo());
}











