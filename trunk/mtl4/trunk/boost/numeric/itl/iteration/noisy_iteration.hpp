// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_NOISY_ITERATION_INCLUDE
#define ITL_NOISY_ITERATION_INCLUDE

#include <iostream>
#include <complex>
#include <string>

#include <boost/numeric/itl/iteration/basic_iteration.hpp>

namespace itl {

  template <class Real>
  class noisy_iteration : public basic_iteration<Real> {
    typedef basic_iteration<Real> super;
  public:
  
    template <class Vector>
    noisy_iteration(const Vector& b, int max_iter_, 
		    Real tol_, Real atol_ = Real(0))
      : super(b, max_iter_, tol_, atol_) { }

    template <class Vector>
    bool finished(const Vector& r) {
      using std::cout;
      using std::endl;

      Real normr_ = std::abs(two_norm(r)); 
      bool ret;
      if (this->converged(normr_))
	ret = true;
      else if (this->i < this->max_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
      cout << "iteration " << this->i << ": resid " 
           << this->resid()
	   << endl;
      return ret;
    }

  
    bool finished(const Real& r) {
      using std::cout;
      using std::endl;

      bool ret;
      if (this->converged(r))
	ret = true;
      else if (this->i < this->max_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
      cout << "iteration " << this->i  << ": resid " 
           << this->resid()
	   << endl;
      return ret;
    }

    template <typename T>
    bool finished(const std::complex<T>& r) { //for the case of complex
      using std::cout;
      using std::endl;

      bool ret;
      if (this->converged(std::abs(r)))
	ret = true;
      else if (this->ii < this->imax_iter)
	ret = false;
      else {
	this->error = 1;
	ret = true;
      }
     cout << "iteration " << this->i << ": resid " 
           << this->resid() << endl;
      return ret;
    }
  
    int error_code() {
      using std::cout;
      using std::endl;

      cout << "finished! error code = " << this->error << endl;
      cout << this->iterations() << " iterations" << endl;
      cout << this->resid() << " is actual final residual. " << endl
	   << this->resid()/this->normb() << " is actual relative tolerance achieved. "
	   << endl;
      cout << "Relative tol: " << this->rtol_ << "  Absolute tol: " << this->atol_ << endl;
      cout << "Convergence:  " << pow(this->rtol_, 1.0 / double(this->iterations())) << endl;
      return this->error;
    }

  };



} // namespace itl

#endif // ITL_NOISY_ITERATION_INCLUDE
