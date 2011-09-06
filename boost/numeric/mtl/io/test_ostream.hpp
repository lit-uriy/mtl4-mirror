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

#ifndef MTL_TEST_OSTREAM_INCLUDE
#define MTL_TEST_OSTREAM_INCLUDE

#include <ostream>

namespace mtl { namespace io {

/// ostream class whose objects only write if MTL_VERBOSE_TEST is defined
struct test_ostream 
{
    /// Constructor for out or std::cout
    test_ostream(std::ostream& out = std::cout) : out(out) {} 

#ifdef MTL_VERBOSE_TEST

    template <typename T>
    test_ostream& operator<<(const T& v)
    {
	out << v;
	return *this;
    }

    void flush() { out.flush(); }
#else
    /// Print on outstream
    template <typename T> test_ostream& operator<<(const T&) { return *this; }
    /// Flush output
    void flush() {}
#endif

  private:
    std::ostream&            out;
};

/// Output stream that writes if MTL_VERBOSE_TEST is defined
static test_ostream tout;

}} // namespace mtl::io

#endif // MTL_TEST_OSTREAM_INCLUDE
