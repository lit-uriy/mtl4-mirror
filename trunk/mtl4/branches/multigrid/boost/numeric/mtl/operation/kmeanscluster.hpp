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

#ifndef MTL_MATRIX_KMEANSCLUSTER_INCLUDE
#define MTL_MATRIX_KMEANSCLUSTER_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/operation/matrep.hpp>
#include <boost/numeric/mtl/operation/dist_matrix.hpp>
#include <boost/numeric/mtl/operation/min_pos.hpp>
#include <boost/numeric/mtl/operation/find.hpp>
#include <boost/numeric/mtl/operation/mean.hpp>


namespace mtl { namespace matrix {

///Returns Original matrix + one col. In the last column are the corresponding k clusters of points
template <typename Matrix>
Matrix inline kmeanscluster(const Matrix& A, const unsigned k)
{
    using mtl::iall;
    typedef typename Collection<Matrix>::size_type  size_type;
    typedef typename Collection<Matrix>::value_type value_type;
    size_type      row, col;

    row= num_rows(A);    col= num_cols(A);
    if (row <= k ) throw mtl::logic_error("more clusters as input");
    irange r(0, imax), kr(0, k), r1(0, imax-1);

    Matrix 	Y(row, col+1), C(k, col),  //C -> sequential init: first point is in first cluster,...  TODO
		D(row,col);
   
    C=A[kr][r];

    for (size_type i = 0; i < col; i++){
	Y[r][i]= A[r][i];//submatrix andersherum??	
    }
    std::cout<< "Y=" << Y << "\n";
//     std::cout<< "C=\n" << C << "\n";
    dense_vector<value_type> tmp(row), z(row), g(row);
    tmp= 0; z= 0; g= 0;
    int i=0;
    //clustering
    while (1) {
	D=dist_matrix(A, C);
	//[z,d]=min(d,[],2)   col_vector z is minimum in row d
			//    col_vector d is position of minimum
	for (size_type j= 0; j < row; j++){
		z[j]= min(D[j][r]);
  		g[j]= min_pos(D[j][r]);
	}

	//if (g == tmp) break; //stop iteration
	//vector equal vector???
	bool yes=true;
	for (size_type j= 0; j < row; j++){
	    if (g[j]==tmp[j]) {
		yes= yes & true;
	    } else {
		yes= yes & false;
	    }
	}
	
	if (yes){
	    break;
	} else {
	    tmp= g;
	}

	for (size_type j= 0; j < k; j++) {
	    dense_vector<unsigned> f(find(g,j));
// 	    std::cout<< "f=" << f << "\n";
	    if (size(f) > 0) {
		dense2D<value_type> M(size(f), col);
		for (size_type k= 0; k < size(f); k++) {
			M[k][r1]= A[f(k)][r1];
		}
		dense_vector<value_type> me(col);
		me= mean(M);

		// c(i,:)=mean(m(find(g==i),:),1);
		C[j][r]=trans(me);
// 		std::cout<< "C=" << C << "\n";
// 		std::cout<< "A=" << A << "\n";
	    }
	}
	
	
	i++;
    }
Y[iall][col]= g;

    return Y;
}
}} // namespace mtl::matrix


#endif // MTL_MATRIX_KMEANSCLUSTER_INCLUDE
