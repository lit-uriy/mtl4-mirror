// Copyright 2006. Peter Gottschling, Matthias Troyer, Rolf Bonderer
// $COPYRIGHT$

#ifndef LA_VECTOR_CONCEPTS_INCLUDE
#define LA_VECTOR_CONCEPTS_INCLUDE


#include <boost/numeric/linear_algebra/concepts.hpp>

#ifdef __GXX_CONCEPTS__

#include <boost/numeric/linear_algebra/ets_concepts.hpp>

namespace math {  
  
concept VectorSpace<typename Vector, typename Scalar = typename Vector::value_type>
: AdditiveAbelianGroup<Vector>
{
    requires Field<Scalar>;
    requires Multiplicable<Scalar, Vector>;
    requires MultiplicableWithAssign<Vector, Scalar>;
    requires DivisibleWithAssign<Vector, Scalar>;
  
    requires std::Assignable<Vector, Multiplicable<Scalar, Vector>::result_type>;
    requires std::Assignable<Vector, Multiplicable<Vector, Scalar>::result_type>;
    requires std::Assignable<Vector, Divisible<Vector, Scalar>::result_type>;
    
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


concept Norm<typename N, typename Vector, 
	     typename Scalar = typename Vector::value_type>
  : std::Callable1<N, Vector>
{
    requires VectorSpace<Vector, Scalar>;
    requires RealMagnitude<Scalar>;
    typename magnitude_type = MagnitudeType<Scalar>::type;
    requires std::Convertible<magnitude_type, Scalar>;

    typename result_type_norm = std::Callable1<N, Vector>::result_type;
    requires std::Convertible<result_type_norm, RealMagnitude<Scalar>::magnitude_type>;
    requires std::Convertible<result_type_norm, Scalar>;

    // Version with function instead functor, as used by Rolf and Matthias
    // Axioms there defined without norm functor and concept has only 2 types
#if 0       
    typename result_type_norm; 
    result_type_norm norm(const Vector&);
    requires std::Convertible<result_type_norm, magnitude_type>;
    requires std::Convertible<result_type_norm, Scalar>;
#endif

    axiom Positivity(N norm, Vector v, magnitude_type ref)
    {
	norm(v) >= zero(ref);
    }

    // The following is covered by RealMagnitude
    // requires AbsApplicable<Scalar>;
    // requires std::Convertible<AbsApplicable<Scalar>::result_type, magnitude_type>;
    // requires Multiplicable<magnitude_type>;

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
    axiom PositiveDefiniteness(N norm, Vector v, magnitude_type ref)
    {
	if (norm(v) == zero(ref))
	    v == zero(v);
	if (v == zero(v))
	    norm(v) == zero(ref);
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
    // requires VectorSpace<Vector, Scalar>;
    requires std::Callable2<I, Vector, Vector>;

    // Result of the inner product must be convertible to Scalar
    requires std::Convertible<std::Callable2<I, Vector, Vector>::result_type, Scalar>;

    // Let's try without this
    // requires ets::InnerProduct<I, Vector, Scalar>;

    requires HasConjugate<Scalar>;

    axiom ConjugateSymmetry(I inner, Vector v, Vector w)
    {
	inner(v, w) == conj(inner(w, v));
    }

    axiom SequiLinearity(I inner, Scalar a, Scalar b, Vector u, Vector v, Vector w)
    {
	inner(v, b * w) == b * inner(v, w);
	inner(u, v + w) == inner(u, v) + inner(u, w);
	// This implies the following (will compilers infere/deduce?)
	inner(a * v, w) == conj(a) * inner(v, w);
	inner(u + v, w) == inner(u, w) + inner(v, w);
    }

    requires RealMagnitude<Scalar>;
    typename magnitude_type = RealMagnitude<Scalar>::type;
    // requires FullLessThanComparable<magnitude_type>;

    axiom NonNegativity(I inner, Vector v, MagnitudeType<Scalar>::type magnitude)
    {
	// inner(v, v) == conj(inner(v, v)) implies inner(v, v) is real
	// ergo representable as magnitude type
	magnitude_type(inner(v, v)) >= zero(magnitude)
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
// Questionable if we want such a concept

#if 0
concept DotProduct<typename I, typename Vector, 
		   typename Scalar = typename Vector::value_type>
  : InnerProduct<I, Vector, Scalar>
{};
#endif

#endif  // __GXX_CONCEPTS__

// Norm induced by inner product
// Might be moved to another place later
// Definition as class and function
// Conversion from scalar to magnitude_type is covered by norm concept
template <typename I, typename Vector,
	  typename Scalar = typename Vector::value_type>
  _GLIBCXX_WHERE(InnerProduct<I, Vector, Scalar> 
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

#ifdef __GXX_CONCEPTS__


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

#endif // __GXX_CONCEPTS__

} // namespace math

#endif // LA_VECTOR_CONCEPTS_INCLUDE
