// $COPYRIGHT$

#ifndef LA_VECTOR_CONCEPTS_INCLUDE
#define LA_VECTOR_CONCEPTS_INCLUDE


#include <boost/numeric/linear_algebra/concepts.hpp>


namespace math {


#ifdef LA_WITH_CONCEPTS


// I'm not sure if we want the division here
concept VectorSpace<typename Vector, typename Scalar = typename Vector::value_type>
: AdditiveAbelianGroup<Vector>
{
    where Field<Scalar>;
    where Multiplicable<Scalar, Vector>;
    where MultiplicableWithAssign<Vector, Scalar>;
    where DivisibleWithAssign<Vector, Scalar>;
  
    where std::Assignable<Vector, Multiplicable<Scalar, Vector>::result_type>;
    where std::Assignable<Vector, Multiplicable<Vector, Scalar>::result_type>;
    where std::Assignable<Vector, Divisible<Vector, Scalar>::result_type>;
    
    // Associated types of Field<Scalar> and AdditiveAbelianGroup<Vector> collide
    // typename result_type = AdditiveAbelianGroup<Vector>::result_type;
    // typename assign_result_type = AdditiveAbelianGroup<Vector>::assign_result_type;

    axiom Distributivity(Vector v, Vector w, Scalar a, Scalar b)
    {
	a * (v + w) == a * v + a * w;
	(a + b) * v == a * v + b * v;
	// The following properties are implied by the above, Field and Abelian group
	// Can we be sure that compilers can deduce/interfere it?
	(v + w) * a == v * a + w * a;
	v * (a + b) == v * a + v * b;
    }
}


// The following concept introduces operations that are not needed in VectorSpace
// but which are very common in numeric software
concept ExtendedVectorSpace<typename Vector, typename Scalar = typename Vector::value_type>
  : VectorSpace<Vector, Scalar>
{
    // valid expression: "vector2 += scalar*vector1"
    typename res_type_1;
    res_type_1 operator+=(Vector&, Multiplicable<Scalar, Vector>::result_type);
    
    // valid expression: "vector2 -= scalar*vector1"
    typename res_type_2;
    res_type_2 operator-=(Vector&, Multiplicable<Scalar, Vector>::result_type);

    // These two epxressions might not be needed with approbriate VectorSpace dealing with ET
};


concept Norm<typename N, typename Vector, 
	     typename Scalar = typename Vector::value_type>
  : std::Callable1<N, Vector>
{
    where VectorSpace<Vector, Scalar>;
    where RealMagnitude<Scalar>;
    typename magnitude_type = MagnitudeType<Scalar>::type;

    typename result_type = std::Callable1<N, Vector>::result_type;
    where std::Convertible<result_type, magnitude_type>;
    where std::Convertible<magnitude_type, Scalar>;

    axiom Positivity(N norm, Vector v, magnitude_type ref)
    {
	norm(v) >= zero(ref);
    }

    // The following is covered by RealMagnitude
    // where AbsApplicable<Scalar>;
    // where std::Convertible<AbsApplicable<Scalar>::result_type, magnitude_type>;
    // where Multiplicable<magnitude_type>;

    axiom PositiveHomogeneity(N norm, Vector v, Scalar a)
    {
	norm(a * v) == abs(a) * norm(v);
    }

    axiom TriangleInequality(N norm, Vector u, Vector v)
    {
	norm(u + v) <= norm(u) + norm(v);
    }
}


concept SemiNorm<typename N, typename Vector, 
		 typename Scalar = typename Vector::value_type>
  : Norm<N, Vector, Scalar>
{
    axiom PositiveDefiniteness(N norm, Vector v)
    {
#if 0
	// axioms with if not yet supported, zero(v) NYD
	if (norm(v) == magnitude_type(0))
	    v == zero(v);
	if (v == zero(v))
	    norm(v) == magnitude_type(0);
#endif
    }
}


// A Banach space is a vector space with a norm
// The (expressible) requirements of Banach Space are already given in Norm.
// The difference between the requirements is the completeness of the 
// Banach space, i.e. that every Cauchy sequence w.r.t. norm(v-w) has a limit
// in the space. Unfortunately, completeness is never satisfied for
// finite precision arithmetic types.
// Another subtle difference is that  Norm is not refined from Vectorspace
concept BanachSpace<typename N, typename Vector, 
		    typename Scalar = typename Vector::value_type>
  : Norm<N, Vector, Scalar>,
    VectorSpace<Vector, Scalar>
{};


concept InnerProduct<typename I, typename Vector, 
		     typename Scalar = typename Vector::value_type>
{
    where VectorSpace<Vector, Scalar>;
    where std::Callable2<I, Vector, Vector>;

    // Result of the inner product must be convertible to Scalar
    where std::Convertible<std::Callable2<I, Vector, Vector>::result_type, Scalar>;

    where ets::InnerProduct<I, Vector, Scalar>;

    where HasConjugate<Scalar>;

    axiom ConjugateSymmetry(I inner, Vector v, Vector w)
    {
	inner(v, w) == conj(inner(w, v));
    }

    axiom SequiLinearity(I inner, Scalar a, Scalar b, Vector u, Vector v, Vector w)
    {
	inner(v, b * w) == b * inner(v, w);
	inner(u, v + w) == inner(u, v) + inner(u, w);
	// This implies the following (will compilers interfere/deduce?)
	inner(a * v, w) == conj(a) * inner(v, w);
	inner(u + v, w) == inner(u, w) + inner(v, w);
    }

    where RealMagnitude<Scalar>;
    typename magnitude_type = RealMagnitude<Scalar>::type;
    // where FullLessThanComparable<magnitude_type>;

    axiom NonNegativity(I inner, Vector v, MagnitudeType<Scalar>::type magnitude)
    {
	// inner(v, v) == conj(inner(v, v)) implies inner(v, v) is real
	// ergo representable as magnitude type
	const_cast<magnitude_type> (inner(v, v)) >= zero(magnitude)
    }

    axiom NonDegeneracy(I inner, Vector v, Vector w, Scalar s)
    {
	if (v == zero(v))
	    inner(v, w) == zero(s);
	if (inner(v, w) == zero(s))
	    v == zero(v);
    }
};


// A dot product is only a semantically special case of an inner product
concept DotProduct<typename I, typename Vector, 
		   typename Scalar = typename Vector::value_type>
  : InnerProduct<I, Vector, Scalar>
{};

#endif  // LA_WITH_CONCEPTS

// Norm induced by inner product
// Might be moved to another place later
// Definition as class and function
// Conversion from scalar to magnitude_type is covered by norm concept
template <typename I, typename Vector,
	  typename Scalar = typename Vector::value_type>
  LA_WHERE(InnerProduct<I, Vector, Scalar> 
	   && RealMagnitude<Scalar>)
struct induced_norm_t
{
    // Return type evtl. with macro to use concept definition
    typename magnitude_type_trait<Scalar>::type
    operator() (const I& inner, const Vector& v)
    {
	// Check whether inner product is positive real
	// assert(Scalar(abs(inner(v, v))) == inner(v, v));
	
	// Similar check while accepting small imaginary values
	// assert( (abs(inner(v, v)) - inner(v, v)) / abs(inner(v, v)) < 1e-6; )
	
	// Could also be defined with abs but that might introduce extra ops
	// typedef RealMagnitude<Scalar>::type magnitude_type;

	typedef typename magnitude_type_trait<Scalar>::type magnitude_type;
	return sqrt(static_cast<magnitude_type> (inner(v, v)));
    }
};


#if 0
template <typename I, typename Vector,
	  typename Scalar = typename Vector::value_type>
  LA_WHERE( InnerProduct<I, Vector, Scalar> 
	    && RealMagnitude<Scalar> )
magnitude_type_trait<Scalar>::type
induced_norm(const I& inner, const Vector& v)
{
    return induced_norm_t<I, Vector, Scalar>() (inner, v);
}
#endif

#ifdef LA_WITH_CONCEPTS


concept HilbertSpace<typename I, typename Vector,
		     typename Scalar = typename Vector::value_type, 
		     typename N = induced_norm_t<I, Vector, Scalar> >
  : InnerProduct<I, Vector, Scalar>,
    BanachSpace<N, Vector, Scalar>
{
    axiom Consistency(Vector v)
    {
	math::induced_norm_t<I, Vector, Scalar>()(v) == N()(v);                    
    }   
};

#endif // LA_WITH_CONCEPTS

} // namespace math

#endif // LA_VECTOR_CONCEPTS_INCLUDE
