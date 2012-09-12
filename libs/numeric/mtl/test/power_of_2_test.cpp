#include <boost/numeric/meta_math/power_of_2.hpp>

#include <cassert>

void test_power_of_2() {
  using namespace meta_math;

  assert( (power_of_2<0>::value==1) );
  assert( (power_of_2<1>::value==2) );
  assert( (power_of_2<2>::value==4) );
  assert( (power_of_2<3>::value==8) );
}


int main() {

  test_power_of_2();

  // all tests passed
  return 0;
}