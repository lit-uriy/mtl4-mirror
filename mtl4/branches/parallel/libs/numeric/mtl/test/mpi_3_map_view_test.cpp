// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.


#include <iostream>


#if defined(MTL_HAS_MPI)

#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <complex>

#include <boost/numeric/mtl/mtl.hpp>

#include <parmetis.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/complex.hpp>

namespace mpi = boost::mpi;

typedef std::complex<double>           ct;

double value(double v) {   return v; }
ct value(ct v) { return ct(real(v), 1.0); }

// scaled value
double svalue(double v) { return 22.0; }
ct svalue(ct v) { return ct(22.0, 2.0); }

// conjugated value
double cvalue(double) { return 22.0; }
ct cvalue(ct) { return ct(22.0, -2.0); }

// complex scaled value
ct csvalue(double) { return ct(0.0, 11.0); }
ct csvalue(ct) { return ct(-1.0, 11.0); }


template <typename Inserter>
struct ins
{
    typedef typename Inserter::size_type  size_type;
    typedef typename Inserter::value_type value_type;

    ins(Inserter& i, int start) : i(i), v(start) {}
    void operator()(size_type r, size_type c) {	i[r][c] << value_type(v++);  }
    Inserter& i;
    int       v;
};

template <typename Matrix>
inline void cv(const Matrix& A, unsigned r, unsigned c, double v)
{
    //if (A[r][c] != v) throw "Wrong value;";
}

template <typename Matrix>
void test(Matrix& A,  const char* name)
{
    using mtl::matrix::agglomerate;
    typedef typename mtl::Collection<Matrix>::size_type  size_type;
    typedef typename mtl::Collection<Matrix>::value_type value_type;
    typedef std::pair<size_type, size_type>              entry_type;
    typedef std::vector<entry_type>                      vec_type;

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    std::vector<idxtype> part;
    mpi::communicator comm(communicator(A));
    int rank= comm.rank();

    A= 0;
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins, 10*(comm.rank()+1));
	switch (comm.rank()) {
	  case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); 
	      part.push_back(1); part.push_back(0); part.push_back(1); break;
	  case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); 
	      part.push_back(0); part.push_back(0); break;
	  case 2: i(5, 6); i(6, 4); i(6, 5);
	      part.push_back(2); part.push_back(2);
	}; 
    }

    sout << "Matrix is:\n" << A;

    mtl::matrix::scaled_view<double, Matrix>  scaled_matrix(2.0, A);
    Matrix scaled_matrix_copy(scaled_matrix);
    sout << "matrix scaled (in distributed form) with 2.0\n" << scaled_matrix_copy << "\n";
    sout << "matrix scaled with 2.0\n" << agglomerate<Matrix>(scaled_matrix) << "\n";

    if (rank == 0 && local(scaled_matrix)[0][2] != svalue(value_type()))
	throw "error in scaling";

    mtl::matrix::conj_view<Matrix>  conj_matrix(A);
    sout << "conjugated matrix\n" << agglomerate<Matrix>(conj_matrix) << "\n";

    if (rank == 0 && local(conj_matrix)[0][2] != cvalue(value_type()))
	throw "error in conjugating";

    typedef  mtl::matrix::distributed<mtl::matrix::compressed2D<ct> >     dist_compl;

    mtl::matrix::scaled_view<ct, Matrix>  cscaled_matrix(ct(0.0, 1.0), A);
    sout << "matrix scaled with i (complex(0, 1))\n" << agglomerate<dist_compl>(cscaled_matrix) << "\n";

    if (rank == 0 && local(cscaled_matrix)[0][2] != csvalue(value_type()))
	throw "error in scaling with i";

#if 0
    mtl::matrix::hermitian_view<Matrix>  hermitian_matrix(A);
    sout << "Hermitian matrix (conjugate transposed)\n" << agglomerate<Matrix>(hermitian_matrix) << "\n";

    if (rank == 0 && local(conj_matrix)[2][0] != cvalue(value_type()))
	throw "error in conjugating";
#endif

    mtl::matrix::rscaled_view<Matrix, double>  rscaled_matrix(A, 2.0);
    sout << "matrix  right scaled with 2.0\n" << agglomerate<Matrix>(rscaled_matrix) << "\n";

    if (rank == 0 && local(rscaled_matrix)[0][2] != svalue(value_type()))
	throw "error in scaling from right";

    mtl::matrix::divide_by_view<Matrix, double>  div_matrix(A, 0.5);
    sout << "matrix divide by 0.5\n" << agglomerate<Matrix>(div_matrix) << "\n";

    if (rank == 0 && local(div_matrix)[0][2] != svalue(value_type()))
	throw "error in scaling from right";
}


int main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(7, 7);
    matrix::distributed<matrix::compressed2D<ct> >     B(7, 7);
    matrix::distributed<matrix::dense2D<double> >      C(7, 7);

    test(A, "compressed2D<double>");
#if 0
    test(B, "compressed2D<complex<double> >");
    test(C, "dense2D<double>");
#endif

    std::cout << "\n**** no errors detected\n";
    return 0;
}

 
#else 

int main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_MPI (and of course"
	      << " the presence of MPI).\n";
    return 0;
}

#endif












