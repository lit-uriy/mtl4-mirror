#include <iostream>

#include "tuple.h"

// ----------------------------------------------------------------------------
// helpers 
// ----------------------------------------------------------------------------

class A {}; 
class B {}; 
class C {};

// classes with different kinds of conversions
class AA {};
class BB : public AA {}; 
struct CC { CC() {} CC(const BB&) {} };
struct DD { operator CC() const { return CC(); }; };

// something to prevent warnings for unused variables
//template<class T> void dummy(const T&) {}

// no public default constructor
class foo {
public:
  explicit foo(int v) : val(v) {}

  bool operator==(const foo& other) const  {
    return val == other.val;
  }

private:
  foo() {}
  int val;
};

// another class without a public default constructor
class no_def_constructor {
  no_def_constructor() {}
public:
  no_def_constructor(std::string) {}
};

// A non-copyable class 
class no_copy {
  no_copy(const no_copy&) {}
public: 
  no_copy() {};
};

// ----------------------------------------------------------------------------
// Testing different element types --------------------------------------------
// ----------------------------------------------------------------------------

using std::tuple;

typedef tuple<int> t1;

typedef tuple<double&, const double&, const double, double*, const double*> t2;
typedef tuple<A, int(*)(char, int), C> t3;
typedef tuple<std::string, std::pair<A, B> > t4;
typedef tuple<A*, tuple<const A*, const B&, C>, bool, void*> t5;
typedef tuple<volatile int, const volatile char&, int(&)(float) > t6;

typedef tuple<B(A::*)(C&), A&> t7;

typedef int(a_function_type)();
typedef void(b_function_type)(int, float);

// An object of this type of tuple cannot be created but the type can
typedef tuple<a_function_type, b_function_type, void, char[10]> t8;

This gives an error:
//typedef t8::tail_type::tail::type::head_type h;

// -----------------------------------------------------------------------
// -tuple construction tests ---------------------------------------------
// -----------------------------------------------------------------------
void test_constructors() {

  no_copy y;
  tuple<no_copy&> x = tuple<no_copy&>(y); // ok
  (void)x;

  char cs[10];
  tuple<char(&)[10]> v2(cs);  // ok
  (void)v2;


}

void test_assignments() {
  using std::tuple;
  using std::make_tuple;
  tuple<float, double>::base t;
  t = make_tuple('a', 5.5);
}


int main() {
  test_constructors();
  test_assignments();
}




// ----------------------------------------------------------

#if 0

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


#endif








