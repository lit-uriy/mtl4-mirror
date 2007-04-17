// $COPYRIGHT$

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

namespace mtl {

/**
 * \defgroup Concepts Concepts
 */
/*@{*/

#ifdef __GXX_CONCEPTS__

    using std::Addable;
    using std::Subtractable;
    using std::Multiplicable;

    auto concept Divisible<typename T, typename U = T>
    {
	typename result_type;
	result_type operator/(const T& t, const U& u);
    };

#else // without concepts

    // Use Joel de Guzman's return type deduction
    // Adapted from uBLAS
    // Differences: 
    //   - Separate types for all operations
    //   - result_type like in concept

    /// Concept Addable: Binary operation
    /** In concept-free compilations also used for return type deduction */ 
    template<class X, class Y>
    class Addable
    {
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
    };


    /// Concept Subtractable: Binary operation
    /** In concept-free compilations also used for return type deduction */ 
    template<class X, class Y>
    class Subtractable
    {
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
    };

    /// Concept Multiplicable: Binary operation
    /** In concept-free compilations also used for return type deduction */ 
    template<class X, class Y>
    class Multiplicable
    {
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
    };
        
    /// Concept Divisible: Binary operation
    /** In concept-free compilations also used for return type deduction */ 
    template<class X, class Y>
    class Divisible
    {
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
	/// Result of division
        typedef typename id::type result_type;
    };
        
#endif


#ifdef __GXX_CONCEPTS__

    concept UnaryStaticFunctor<typename T>
    {
	typename result_type;
	
	static result_type apply(T);
    };

#else

    /// Concept UnaryStaticFunctor
    /**
       \par Refinement of:
       - BinaryStaticFunctor <T, U>
    */
    template <typename T>
    struct UnaryStaticFunctor
	: public UnaryFunctor
    {
	/// Result type of apply
	typedef associated_type result_type;
	
	/// The unary static function
	static result_type apply(T);

	/// The application operator behaves like apply. Exists for compatibility with UnaryFunctor
	result_type operator()(T);
    };
#endif


#ifdef __GXX_CONCEPTS__

    concept BinaryStaticFunctor<typename T, typename U>
    {
	typename result_type;
	
	static result_type apply(T, U);
    };

#else

    /// Concept BinaryStaticFunctor
    /**
       \par Refinement of:
       - BinaryStaticFunctor <T, U>
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

#endif

/*@}*/ // end of group Concepts

} // namespace mtl

#endif // MTL_STD_CONCEPT_INCLUDE
