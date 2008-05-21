// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_BASIC_ITERATION_INCLUDE
#define ITL_BASIC_ITERATION_INCLUDE

#include <iostream>
#include <complex>
#include <string>

namespace itl {


  template <class Real>
  class basic_iteration {
  public:

  
    typedef Real real;

  
    template <class Vector>
    basic_iteration(const Vector& b, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), normb_(std::abs(two_norm(b))), 
	       max_iter(max_iter_), rtol_(t), atol_(a) { }
  
    basic_iteration(Real nb, int max_iter_, Real t, Real a = Real(0))
      : error(0), i(0), normb_(nb), max_iter(max_iter_), rtol_(t), atol_(a) {}

  
    template <class Vector>
    bool finished(const Vector& r) {
      Real normr_ = std::abs(two_norm(r)); 
      if (converged(normr_))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

  
    bool finished(const Real& r) {
      if (converged(r))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

    template <typename T>
    bool finished(const std::complex<T>& r) { 
      if (converged(std::abs(r)))
	return true;
      else if (i < max_iter)
	return false;
      else {
	error = 1;
	return true;
      }
    }

    inline bool converged(const Real& r) {
      if (normb_ == 0)
	return r < atol_;  // ignore relative tolerance if |b| is zero
      resid_ = r / normb_;
      return (resid_ <= rtol_ || r < atol_); // relative or absolute tolerance.
    }

    inline void operator++() { ++i; }
  
    inline bool first() { return i == 0; }
  
    inline int error_code() { return error; }
  
    inline int iterations() { return i; }
  
    inline Real resid() { return resid_ * normb_; }
  
    inline Real normb() const { return normb_; }
  
    inline Real tol() { return rtol_; }
    inline Real atol() { return atol_; } 
  
    inline void fail(int err_code) { error = err_code; }
  
    inline void fail(int err_code, const std::string& msg)
    { error = err_code; err_msg = msg; }
  
    inline void set(Real v) { normb_ = v; }

  protected:
    int error;
    int i;
    const Real normb_;
    int max_iter;
    Real rtol_;
    Real atol_;
    Real resid_;
    std::string err_msg;
  };


} // namespace itl

#endif // ITL_BASIC_ITERATION_INCLUDE
