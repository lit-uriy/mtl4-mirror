
#if 0

#include <iostream>
#include <cmath>
#include <set>


#ifdef __GXX_CONCEPTS__
#  include <concepts>
#  include <boost/numeric/linear_algebra/new_concepts.hpp>
#else 
#  include <boost/numeric/linear_algebra/pseudo_concept.hpp>
#endif

#endif

#define DYNAMIC_CONCEPT(SCONCEPT)		\
    std::set<const void*> table_ ## SCONCEPT;   \
                                                \
    template <typename T>                       \
    bool is_ ## SCONCEPT(const T& x)            \
    { return table_ ## SCONCEPT.find(&x) != table_ ## SCONCEPT.end(); } \
                                                \
    template <typename T>                       \
      requires SCONCEPT<T>                      \
    bool is_ ## SCONCEPT(const T& x)            \
    { return true; }                            \
                                                \
    template <typename T>                       \
    void map_ ## SCONCEPT(const T& x)           \
    { table_ ## SCONCEPT.insert(&x); }          \
                                                \
    template <typename T>                       \
    void unmap_ ## SCONCEPT(const T& x)         \
    { table_ ## SCONCEPT.erase(&x); }          

#define SELECT(CONDITION, F)                    \
    if (is_ ## CONDITION(x)) { F(x); return; }

#define SELECT2(C1, C2, F)				\
    if (is_ ## C1(x) && is_ ## C2(x)) { F(x); return; }

struct mat {};                 // Matrix type
struct smat : public mat {};   // Symmetric matrix type


concept Symmetric<typename Matrix> { /* axioms */ }
concept PositiveDefinit<typename Matrix>{ /* axioms */ }

DYNAMIC_CONCEPT(Symmetric)
DYNAMIC_CONCEPT(PositiveDefinit)

concept_map Symmetric<smat> {}


template <typename Matrix>
void spd_solver(const Matrix& A)
{
    std::cout << "spd_solver (Symmetric positiv-definit)\n";
}

template <typename Matrix>
void symmetric_solver(const Matrix& A)
{
    std::cout << "symmetric_solver\n";
}


template <typename Matrix>
void solver(const Matrix& x)
{
    SELECT2(PositiveDefinit, Symmetric, spd_solver);
    SELECT(Symmetric,                   symmetric_solver);

    std::cout << "Default_solver\n";
}

int main(int, char* [])  
{
    mat  A, B;
    smat C, D;

    map_Symmetric(B);
    map_PositiveDefinit(D);

    solver(A);
    solver(B);
    solver(C);
    solver(D);

    return 0;
}
