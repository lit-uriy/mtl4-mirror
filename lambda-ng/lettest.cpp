#include "env.hpp"
#include "nodes.hpp"
#include "let.hpp"
#include <iostream>

template <class Body>
LAMBDA_AUTO_FUNC(run_prog(const Body& body), body.run(empty_env))

VAR(a);

int main(int, char**) {
  using namespace std;
  cout << run_prog(let(a,5,a)) << endl;
  return 0;
}
