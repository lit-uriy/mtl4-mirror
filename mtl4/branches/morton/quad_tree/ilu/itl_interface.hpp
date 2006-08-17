/*****************************************************************************
  file: itl_interface.hpp
  -----------------------
  Defines functions to be used with the ITL's GMRes function:
  	- mult: w = Ax + b
  	- solve  --> forward and backward subtitutions.

  For now, Make sure to pre-instantiate a global Matrix object called "mat",
  for bounds checking... This object holds all the data descriptive of the
  original matrix, that was destructively LU decomposed and whose quadtree
  representation's head node was passed into the solver (w/ type name Both)

  Created on: 08/13/06

  Larisse D. Voufo
  Peter Gottschling
*****************************************************************************/
#ifndef ITL_INTERFACE_HPP
#define ITL_INTERFACE_HPP

#include <vector>
#include "LU.h"

//------------- Function Abstractions -------------------

/*
Matrix-vector multiply
produces: w = Ax + b
*/
//__attribute__((always inline))
template <typename Vx, typename Vb, typename Vw>
inline void mult( const Both& A, const Vx& x, const Vb& b, Vw& w)
{
	mult( A, MTN_START, BND_PART_ALL, LEVEL_START, x, 0, b, w, 0);
}


/*
sys. solver:
Ax = b.
produce x.
forward + backward substitution.
*/
// __attribute__((always inline))
template <typename Vb, typename Vx>
inline void solve(const Both& M, const Vb& b, Vx& x);

#include "itl_interface_impl.hpp"

#endif  // ITL_INTERFACE_HPP

/////////////////////////////////////////////////////////////////////////////
