#include <iostream>

#pragma gcc_extensions on

#include "env.hpp"
#include "nodes.hpp"
#include "let.hpp"

template <class Body>
typeof(wrap(Body()).run(empty_env))
run_prog(const Body& body) {
  return wrap(body).run(empty_env);
}

VAR(a);

int main(int, char**) {
  using namespace std;
  cout << run_prog(5) << endl;
  //cout << run_prog(a) << endl;
  let_type<int, a_,constant_type<int>,constant_type<int> > b;
  cout << run_prog(let<int>(a,5,6)) << endl;
  return 0;
}
