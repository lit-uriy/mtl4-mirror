#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace mtl;
using namespace itl;


template <typename At, typename Lt, typename Ut>
void dense_ilu_0(const At& As, const Lt& Ls, const Ut& Us)
{
    dense2D<double> LU(As);
     
    const int n= num_rows(LU);
    for (int i= 1; i < n; i++) 
	for (int k= 0; k < i; k++) {
	    LU[i][k]/= LU[k][k];
	    for (int j= k + 1; j < n; j++)
		if (LU[i][j] != 0)
		    LU[i][j]-= LU[i][k] * LU[k][j];
	}
    std::cout << "Factorizing A = \n" << As << "-> LU = \n" << LU;    

    if (std::abs(LU[3][2] - Ls[3][2]) > 0.001) 
	throw "Wrong value in L for sparse ILU(0) factorization";

    if (std::abs(LU[3][3] - Us[3][3]) > 0.001)
	throw "Wrong value in U for sparse ILU(0) factorization";
}


int main()
{
    // For a more realistic example set sz to 1000 or larger
    const int size = 1000, N = size * size; 

    typedef compressed2D<double>  matrix_type;
    compressed2D<double>          A(N, N), dia(N, N);
    matrix::laplacian_setup(A, size, size);
    // dia= 1.0; A+= dia;
    
   
    pc::ilu_0<matrix_type>        P(A);
    dense_vector<double>          x(N, 1.0), b(N);
    
    if(size > 1 && size < 4)
	dense_ilu_0(A, P.get_L(), P.get_U());

    b = A * x;
    x= 0;
    
    noisy_iteration<double> iter(b, N, 1.e-6);
    bicgstab(A, x, b, P, iter);
    
    return 0;
}
