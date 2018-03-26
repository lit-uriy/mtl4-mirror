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

#ifndef MTL_STD_CONCEPT_INCLUDE
#define MTL_STD_CONCEPT_INCLUDE

#ifdef __GXX_CONCEPTS__
#  include <concepts>
#else
// Use Joel de Guzman's return type deduction
#  include <boost/numeric/ublas/detail/returntype_deduction.hpp>
#  include <boost/mpl/at.hpp>
#  include <boost/numeric/linear_algebra/pseudo_concept.hpp>
#endif

// #include <boost/numeric/mtl/utility/is_what.hpp>         -> leads to cyclic inclusions
// #include <boost/numeric/mtl/operation/mult_result.hpp>

namespace mtl {

/**
 * \defgroup Concepts Concepts
 */
/*@{*/


    // Use Joel de Guzman's return type deduction
    // Adapted from uBLAS
    // Differences: 
    //   - Separate types for all operations
    //   - result_type like in concept

    /// Concept Addable: Binary operation
    /** In concept-free compilations also used for return type deduction.
        If available use decltype instead of meta-programming. */ 
    template<class X, class Y>
    struct Addable
    {
#    ifdef MTL_WITH_AUTO
	/// Result of addition
	typedef decltype(X() + Y())   result_type;
#    else
      private:
        typedef boost::numeric::ublas::type_deduction_detail::base_result_of<X, Y> base_type;
        static typename base_type::x_type x;
        static typename base_type::y_type y;
        static const std::size_t size = sizeof (
                   boost::numeric::ublas::type_deduction_detail::test<
                        typename base_type::x_type
                      , typename base_type::y_type
		   >(x + y)     
                );

        static const std::size_t index = (size / sizeof (char)) - 1;
        typedef typename boost::mpl::at_c<
    	typename base_type::types, index>::type id;
      public:
	/// Result of addition
        typedef typename id::type result_type;
#    endif
    };


    /// Concept Subtractable: Binary operation
    /** In concept-free compilations also used for return type deduction.
        If available use decltype instead of meta-programming. */ 
    template<class X, class Y>
    struct Subtractable
    {
#    ifdef MTL_WITH_AUTO
	/// Result of addition
	typedef decltype(X() - Y())   result_type;
#    else
      private:
        typedef boost::numeric::ublas::type_deduction_detail::base_result_of<X, Y> base_type;
        static typename base_type::x_type x;
        static typename base_type::y_type y;
        static const std::size_t size = sizeof (
                   boost::numeric::ublas::type_deduction_detail::test<
                        typename base_type::x_type
                      , typename base_type::y_type
		   >(x - y)     
                );

        static const std::size_t index = (size / sizeof (char)) - 1;
        typedef typename boost::mpl::at_c<
    	typename base_type::types, index>::type id;
      public:
	/// Result of subtraction
        typedef typename id::type result_type;
#    endif
    };


    /// Concept Multiplicable: Binary operation
    /** In concept-free compilations also used for return type deduction.
        If available use decltype instead of meta-programming.  */ 
    template<class X, class Y>
    struct Multiplicable
    {
#    ifdef MTL_WITH_AUTO
	/// Result of multiplication
	typedef decltype(X() * Y())   result_type;
#    else
      private:
        typedef boost::numeric::ublas::type_deduction_detail::base_result_of<X, Y> base_type;
        static typename base_type::x_type x;
        static typename base_type::y_type y;
        static const std::size_t size = sizeof (
                   boost::numeric::ublas::type_deduction_detail::test<
                        typename base_type::x_type
                      , typename base_type::y_type
                    >(x * y)     
                );

        static const std::size_t index = (size / sizeof (char)) - 1;
        typedef typename boost::mpl::at_c<
    	typename base_type::types, index>::type id;
      public:
	/// Result of multiplication
        typedef typename id::type result_type;
#    endif
    };

    /// Concept Divisible: Binary operation
    /** In concept-free compilations also used for return type deduction.
        If available use decltype instead of meta-programming.  */ 
    template<class X, class Y>
    struct Divisible
    {
#    ifdef MTL_WITH_AUTO
	/// Result of division
	typedef decltype(X() / Y())   result_type;
#    else
      private:
        typedef boost::numeric::ublas::type_deduction_detail::base_result_of<X, Y> base_type;
        static typename base_type::x_type x;
        static typename base_type::y_type y;
        static const std::size_t size = sizeof (
                   boost::numeric::ublas::type_deduction_detail::test<
                        typename base_type::x_type
                      , typename base_type::y_type
                    >(x / y)     
                );

        static const std::size_t index = (size / sizeof (char)) - 1;
        typedef typename boost::mpl::at_c<
    	typename base_type::types, index>::type id;
      public:
	/// Result of division
        typedef typename id::type result_type;
#    endif
    };
        

    /// Concept UnaryFunctor
    /** With concept corresponds to std::Callable1 */ 
    template <typename T>
    struct UnaryFunctor
    {
	/// Result type of operator()
	typedef associated_type result_type;
	
	/// The unary  function
	result_type operator()(T);
    };


    /// Concept UnaryStaticFunctor
    /**
       \par Refinement of:
       - std::Callable1 < T >
    */
    template <typename T>
    struct UnaryStaticFunctor
      : public UnaryFunctor<T>
    {
	/// Result type of apply
	typedef associated_type result_type;
	
	/// The unary static function
	static result_type apply(T);

	/// The application operator behaves like apply. Exists for compatibility with UnaryFunctor
	result_type operator()(T);
    };

    /// Concept BinaryFunctor
    /** With concept corresponds to std::Callable2 */ 
    template <typename T, typename U>
    struct BinaryFunctor
    {
	/// Result type of operator()
	typedef associated_type result_type;
	
	/// The unary  function
	result_type operator()(T, U);
    };

    /// Concept BinaryStaticFunctor
    /**
       \par Refinement of:
       - BinaryFunctor <T, U>
    */
    template <typename T, typename U>
    struct BinaryStaticFunctor
	: public BinaryFunctor <T, U>
    {
	/// Result type of apply
	typedef associated_type result_type;
	
	/// The unary static function
	static result_type apply(T, U);

	/// The application operator behaves like apply. Exists for compatibility with BinaryFunctor
	result_type operator()(T, U);
    };

/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_STD_CONCEPT_INCLUDE
