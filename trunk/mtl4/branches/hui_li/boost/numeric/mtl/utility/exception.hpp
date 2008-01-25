// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MTL_EXCEPTION_INCLUDE
#define MTL_MTL_EXCEPTION_INCLUDE

#include <cassert>
#include <stdexcept>

namespace mtl {

// If MTL_ASSERT_FOR_THROW is defined all throws become assert
// MTL_DEBUG_THROW_IF completely disappears if NDEBUG is defined
#ifndef NDEBUG
#  ifdef MTL_ASSERT_FOR_THROW
#    define MTL_DEBUG_THROW_IF(Test, Exception) \
        assert(!(Test));
#  else
#    define MTL_DEBUG_THROW_IF(Test, Exception) \
        if (Test) throw Exception;
#  endif
#else
#  define MTL_DEBUG_THROW_IF(Test,Exception)
#endif


#ifdef MTL_ASSERT_FOR_THROW
#  define MTL_THROW_IF(Test, Exception)       \
   {                                          \
       assert(!(Test));			      \
   }
#else
#  define MTL_THROW_IF(Test, Exception)       \
   {                                          \
      if (Test) throw Exception;              \
   }
#endif


#ifdef MTL_ASSERT_FOR_THROW
#  define MTL_THROW(Exception)       \
   {                                 \
       assert(0);		     \
   }
#else
#  define MTL_THROW(Exception)       \
   {                                 \
      throw Exception;               \
   }
#endif


#if 0 
standard errors:

exception
    logic_error
        domain_error
        invalid_argument
        length_error
        out_of_range
    runtime_error
        range_error
        overflow_error
        underflow_error
bad_alloc
bad_cast
bad_exception
bad_typeid

#endif

/// Exception for indices out of range
struct index_out_of_range
    : public std::out_of_range
{
    /// Error can be specified more precisely in constructor if desired
    explicit index_out_of_range(const char *s= "Index out of range") 
	: std::out_of_range(s) {}
};

/// Exception for invalid range definitions, esp. in constructors
struct range_error
    : public std::range_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit range_error(const char *s= "Invalid range") : std::range_error(s) {}
};

/// Exception for arguments with incompatible sizes
struct incompatible_size
    : public std::domain_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit incompatible_size(const char *s= "Arguments have incompatible size.")
	: std::domain_error(s) {}
};

/// Exception for arguments with incompatible shapes, e.g. adding matrices and vectors
struct argument_result_conflict
    : public std::domain_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit argument_result_conflict(const char *s= "Used same object illegally as argument and result.")
	: std::domain_error(s) {}
};

/// Exception for arguments with incompatible shapes, e.g. adding matrices and vectors
struct incompatible_shape
    : public std::domain_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit incompatible_shape(const char *s= "Arguments have incompatible shape.")
	: std::domain_error(s) {}
};

/// Exception for arguments with incompatible sizes
struct matrix_not_square
    : public std::domain_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit matrix_not_square(const char *s= "Matrix must be square for this operation.")
	: std::domain_error(s) {}
};

/// Exception for arguments with incompatible sizes
struct not_expected_result
    : public std::domain_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit not_expected_result(const char *s= "The result of an operation is not the expected one.")
	: std::domain_error(s) {}
};

/// Exception for run-time errors that doesn't fit into specific categories
struct runtime_error
    : public std::runtime_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit runtime_error(const char *s= "Run-time error")
	: std::runtime_error(s) {}
};

/// Exception for logic errors that doesn't fit into specific categories
struct logic_error
    : public std::logic_error
{
    /// Error can be specified more precisely in constructor if desired
    explicit logic_error(const char *s= "Logic error")
	: std::logic_error(s) {}
};

} // namespace mtl

#endif // MTL_MTL_EXCEPTION_INCLUDE
