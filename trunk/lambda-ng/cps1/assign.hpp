#ifndef LAMBDA_ASSIGN_HPP
#define LAMBDA_ASSIGN_HPP

#include "tags.hpp"
#include "set.hpp"

template <class Lhs, class Rhs>
struct assign_node_type: public expr_node {
  typedef assign_node_type<Lhs,Rhs> self;
  LAMBDA_NODE_CONTENTS

  Lhs lhs; Rhs rhs;
  assign_node_type(const Lhs& lhs_, const Rhs& rhs_):
    lhs(lhs_), rhs(rhs_) {}

  template <class Env, class K>
  void run(const Env& env, const K& k) const {
    set(lhs,rhs).run(env,k);
  }
};
template <class Lhs, class Rhs>
assign_node_type<typename node_traits<Lhs>::wrapped,
		 typename node_traits<Rhs>::wrapped>
assign_node(const Lhs& lhs, const Rhs& rhs) {
  return assign_node_type<typename node_traits<Lhs>::wrapped,
			  typename node_traits<Rhs>::wrapped>
    (wrap(lhs),wrap(rhs));
}

#endif // LAMBDA_ASSIGN_HPP
