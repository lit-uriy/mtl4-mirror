#include "tuple"
#include <string>
#include <iostream>

int main(int, char**) {
  std_tuple::tuple<int, const int, std::string> a(1, 'a', "Hi");
  std::cout << get<0>(a) << " " << get<1>(a) << " " << get<2>(a) << std::endl;
  return 0;
}
