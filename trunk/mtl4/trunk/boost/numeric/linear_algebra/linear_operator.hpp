// $COPYRIGHT$

#ifndef MATH_LINEAR_OPERATOR_INCLUDE
#define MATH_LINEAR_OPERATOR_INCLUDE

#ifdef __GXX_CONCEPTS__
#  include <concepts>
#endif

namespace math {

#ifdef __GXX_CONCEPTS__
    concept LinearOperator<typename Operator, typename VectorDomain, typename VectorImage>
    {
	requires VectorSpace<VectorDomain>;
	requires VectorSpace<VectorImage>;

	typename result_type;
	result_type operator* (Operator, VectorDomain);
	
	requires Assignable<VectorImage, result_type>;

	// The following two requirements are subject to discussion
	typename plus_assign_type;
	plus_assign_type operator+= (VectorImage, result_type);
	
	typename minus_assign_type;
	minus_assign_type operator+= (VectorImage, result_type);

	axiom Addability(Operator op, VectorDomain x, VectorDomain y)
	{
	    op(x + y) == op(x) + op(y);
	}

	// The two vector spaces must be scalable with the same scalar types
	axiom Scalability(Operator op, VectorSpace<VectorDomain>::scalar_type a, VectorDomain x)
	{
	    op( a * x ) == a * op(x);
	}
    };
#else
    //! Concept LinearOperator
    /*!
        Linear operator from one vector space into another one.

        \param Operator The type of the operator, e.g., some matrix type
	\param VectorDomain The the type of a vector in the domain vector space
	\param VectorImage The the type of a vector in the image vector space
	
        \par Refinement of:
	- VectorSpace <VectorDomain>
	- VectorSpace <VectorImage>

        \par Notation:
        <table summary="notation">
          <tr>
            <td>op</td>
	    <td>Object of type Operation</td>
	  </tr>
          <tr>
            <td>x, y</td>
	    <td>Objects of type VectorDomain</td>
          </tr>
        </table>

        \invariant
        <table summary="invariants">
          <tr>
            <td>Addability</td>
	    <td>op(x + y) == op(x) + op(y)</td>
          </tr>
          <tr>
            <td>Scalability</td>
	    <td>op( a * x ) == a * op(x)</td>
          </tr>
        </table>
	
     */
   template <typename Operator, typename VectorDomain, typename VectorImage>
    struct LinearOperator
    {
	typename result_type;
	result_type operator* (Operator, VectorDomain);

    }
#endif

} // namespace math

#endif // MATH_LINEAR_OPERATOR_INCLUDE
