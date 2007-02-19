#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include "itl_interface.hpp"    //for 
//#include "LU.h"

using namespace mtl;

typedef  double Type;


int main(int argc, char** argv) 
{

  cout << "start\n";
  
  //  Matrix mat;
  env_init(argc, argv);

  mat.printRowMaj(mat.mainNode);
/*
  vector<double>   x(10), b(10);
  fill(b.begin(), b.end(), 3.0);

  cout<<"b: ";
  copy( b.begin(), b.end(), ostream_iterator<double>(cout, " ") );
  cout<<endl; 
*/
  // -- apply gmres --  
  int max_iter = 50;
  
  dense1D<Type> x(mat.getRows(), Type(0));
  dense1D<Type> b(mat.getCols());
  for (dense1D<Type>::iterator i=b.begin(); i!=b.end(); i++)
    *i = 1.;
  
	//LUD
	mat.copyMatrix(&(mat.backupNode), mat.mainNode, LEVEL_START);
  LUD( &(mat.mainNode),
				MTN_START, BND_PART_ALL, BND_PART_ALL , LEVEL_START );
  
  dense1D<Type> b2(mat.getCols());
	solve( LUmat, b, b2);
	
	//iteration
  noisy_iteration<double> iter(b2, max_iter, 1e-6);
  int restart = 10;   //restart 10 times
  
  typedef dense1D<Type> Vec;
  classical_gram_schmidt<Vec> orth(restart, size(x));
  
  //gmres algorithm
     //L = left preconditioner & U= right preconditionner
  //gmres(mat.backupNode, x, b, mat.mainNode, mat.mainNode, restart, iter, orth);
     // left conditiooner = M = L*U & right preconditinner = I
  gmres(mat.backupNode, x, b, mat.mainNode,  restart, iter, orth);
  
  //verify the result
  dense1D<Type> b1(A.ncols());
  mult(A, x, b1);
  itl::add(itl::scaled(b, -1.), b1);

  if ( itl::two_norm(b1) < 1.e-6) { //output
    for (int i=0; i<5; i++) {
      for (int j=0; j<5; j++) {
	cout.width(6);
	//!!! it is not recommended to use A(i,j) for large sparse matrices
	cout << A(i, j) << " ";
      }
      cout << "  x  ";
      cout.width(16);
      cout << x[i] << "  =  ";
      cout.width(6);
      cout << b[i] << endl;
    }
  } else {
    cout << "Residual " << iter.resid() << endl;
  }
  
  //kill matrix environment
  env_kill();
  
  return 0;
}

