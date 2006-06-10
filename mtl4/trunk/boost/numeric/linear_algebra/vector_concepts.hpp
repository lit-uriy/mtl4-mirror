// $COPYRIGHT$

#ifndef LA_VECTOR_CONCEPTS_INCLUDE
#define LA_VECTOR_CONCEPTS_INCLUDE


#include <boost/numeric/linear_algebra/concepts.hpp>


namespace math {


#ifdef LA_WITH_CONCEPTS


// I'm not sure if we want the division here
concept VectorSpace<typename Vector, typename Scalar = Vector::value_type>
  : Field<Scalar>,
    AdditiveAbelianGroup<Vector>,
    Multiplicable<Scalar, Vector>,
    MultiplicableWithAssign<Vector, Scalar>,
    DivisibleWithAssign<Vector, Scalar>
{
    where std::Assignable<Vector, Multiplicable<Scalar, Vector>::result_type>;
    where std::Assignable<Vector, Multiplicable<Vector, Scalar>::result_type>;
    where std::Assignable<Vector, Divisible<Vector, Scalar>::result_type>;
    
    axiom Distributivity(Vector v, Vector w, Scalar a, Scalar b)
    {
	a * (v + w) == a * v + a * w;
	(a + b) * v == a * v + b * v;
	// The following properties are implied by the above, Field and Abelian group
	// Can we be sure that the compiler can deduce it?
	(v + w) * a == v * a + w * a;
	v * (a + b) == v * a + v * b;
    }
}


// The following concept introduces operations that are not needed in VectorSpace
// but which are very common in numeric software
concept ExtendedVectorSpace<typename Vector, typename Scalar = Vector::value_type>
  : VectorSpace<Vector, Scalar>
{
    // valid expression: "vector2 += scalar*vector1"
    typename res_type_1;
    res_type_1 operator+=(Vector&, Multiplicable<Scalar, Vector>::result_type);
    
    // valid expression: "vector2 -= scalar*vector1"
    typename res_type_2;
    res_type_2 operator-=(Vector&, Multiplicable<Scalar, Vector>::result_type);
};


concept Norm<typename Operation, typename Range, typename Image>
  : std::Callable1<Operation, Range>,
    FullEqualityComparable<Image>,
    FullLessThanComparable<Image>
{
    where std::Convertible<result_type, Image>;

    axiom Positivity(Operation n, Range v, Image ref)
    {
	n(v) >= Image(0); // or zero(ref)
    }

    axiom PositiveHomogeneity(Operation n, Range v, Image)
    {
	n(v) >= Image(0); // or zero(ref)
    }
    

}


concept Norm<typename Operation, typename Vector, 
	     typename Scalar = Vector::value_type>
  : std::Callable1<Operation, Vector>
{
    where VectorSpace<Vector, Scalar>;
    where MagnitudeType<Scalar>;
    typename magnitude_type = MagnitudeType<Scalar>::type;

    typename result_type = std::Callable1<Operation, Vector>;
    where std::Convertible<result_type, magnitude_type>;

    axiom Positivity(Operation norm, Vector v, magnitude_type ref)
    {
	norm(v) >= magnitude_type(0); // or zero(ref)
    }

    where AbsApplicable<Scalar>;
    where std::Convertible<AbsApplicable<Scalar>::result_type, magnitude_type>;
    where Multiplicable<magnitude_type>;

    axiom PositiveHomogeneity(Operation norm, Vector v, Scalar a)
    {
	norm(a * v) == abs(a) * norm(v);
    }

    axiom TriangleInequality(Operation norm, Vector u, Vector v)
    {
	norm(u + v) <= norm(u) + norm(v);
    }
}


concept SemiNorm<typename Operation, typename Vector, 
		 typename Scalar = Vector::value_type>
  : Norm<Operation, Vector, Scalar>
{
    axiom PositiveDefiniteness(Operation norm, Vector v)
    {
#if 0
	// axioms with if NYS, zero(v) NYD
	if (norm(v) == magnitude_type(0))
	    v == zero(v);
	if (v == zero(v))
	    norm(v) == magnitude_type(0);
#endif
    }
}

#endif  // LA_WITH_CONCEPTS

}

#endif


#endif // LA_VECTOR_CONCEPTS_INCLUDE
