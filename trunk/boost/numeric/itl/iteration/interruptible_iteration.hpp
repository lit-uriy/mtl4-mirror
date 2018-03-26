// Software License for MTL
//
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
//
// This file is part of the Matrix Template Library
//
// See also license.mtl.txt in the distribution.

#ifndef ITL_INTERRUPTIBLE_ITERATION_INCLUDE
#define ITL_INTERRUPTIBLE_ITERATION_INCLUDE

#include <atomic>
#include "basic_iteration.hpp"

#include <boost/numeric/itl/iteration/basic_iteration.hpp>


namespace itl {

class interruptible_iteration
{
  public:
    typedef double real; // hack  
      
    interruptible_iteration(basic_iteration<double>& iter) 
      : iter(&iter), interrupted(false) {}

    interruptible_iteration() : iter(0), interrupted(false) {}  
    
    void set_iter(basic_iteration<double>& new_iter) { iter= &new_iter; }
      
    template <typename T>
    bool finished(const T& r) { return iter->finished(r) || interrupted.load(); }
    
    void interrupt() { interrupted= true; }
    bool is_interrupted() const { return interrupted.load(); }
    
    operator int() const { return *iter; }

    int error_code() const { return iter->error_code(); }

    interruptible_iteration& operator++() { ++*iter; return *this; }
    
    bool first() const { return iter->first(); }
    
  private:
    basic_iteration<double>*  iter; // not owned only referred
    std::atomic<bool>         interrupted;
};

} // namespace itl

#endif // ITL_INTERRUPTIBLE_ITERATION_INCLUDE
