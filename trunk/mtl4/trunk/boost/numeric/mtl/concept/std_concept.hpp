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
	typedef typename id::type result_type;
    };


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
        typedef typename id::type result_type;
    };

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
        typedef typename id::type result_type;
    };
        
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
    template <typename T>
    struct UnaryStaticFunctor
    {
	/// Result type of apply
	typedef associated_type result_type;
	
	/// The unary static function
	static result_type apply(T);
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
    template <typename T, typename U>
    struct BinaryStaticFunctor
    {
	/// Result type of apply
	typedef associated_type result_type;
	
	/// The unary static function
	static result_type apply(T, U);
    };

#endif


} // namespace mtl

#endif // MTL_STD_CONCEPT_INCLUDE
