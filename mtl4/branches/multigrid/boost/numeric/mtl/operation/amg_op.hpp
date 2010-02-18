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

// Written by Jan Bos
// Edited  by Peter Gottschling

#ifndef MTL_MATRIX_AMG_OP_INCLUDE
#define MTL_MATRIX_AMG_OP_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/matrix/strict_upper.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/max.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mtl { namespace matrix {

///first num_cols(A)/2 values are fine grid values, the rest is coarse
template < typename Matrix, typename Vector>
Vector inline amg_coarsepoints_default(const Matrix& A, Vector& b)
{
   typedef typename mtl::Collection<Matrix>::value_type Scalar;
   typedef typename mtl::Collection<Matrix>::size_type  Size;
   Size n = num_cols(A), k=Size(floor(n/2));

   if (n <= 2) throw mtl::logic_error("to small to coarse");

    Vector c(n), f(n);
    c= 1;  // all points in coarse grid
    f= 0; // no points in exclude of c
	for (Size i = 0; i < k; i++) {
		f[i]= 1;
	}
	c= c - f;
    return c;
}

///chose the largest absolut value of Offdiagonal. The row index is classified as fine
///the colum index as coarse
template < typename Matrix, typename Vector >
Vector inline amg_coarsepoints_simple(const Matrix& A, Vector& b)
{
   typedef typename mtl::Collection<Matrix>::value_type Scalar;
   typedef typename mtl::Collection<Matrix>::size_type  Size;
   Size n = num_cols(A), i, j;
   dense2D<Scalar> B(A);

   if (n <= 2) throw mtl::logic_error("to small to coarse");

   Vector c(n), f(n), u(n); 
   c= 0; // no points in coarse grid
   f= 0; // no points in exclude of c
   u= 1; // all Points
	
	while (sum(u) >= 1) {
		boost::tie(i,j)= max_abs_pos(B);
		B[i][j]= 0;
		if( (i != j) && (u[i]==1) && (u(j)==1)) {
			f[i]= 1;
			c[j]= 1;
			u[i]= 0;
			u[j]= 0;
		}
	}
    return c;
}



///chose the largest absolut value of Offdiagonal. The row index is classified as fine
///the colum index as coarse
template < typename Matrix, typename Vector >
Vector inline amg_coarsepoints_rs(Matrix& A, Vector& b, double beta)
{
   using mtl::imax; using std::abs;
   typedef typename mtl::Collection<Matrix>::value_type Scalar;
   typedef typename mtl::Collection<Matrix>::size_type  Size;
   Size m = num_cols(A), n = num_rows(A), i, j, argmax, ialt=0;
   Matrix B(A);
   mtl::multi_vector<Vector>         S(n, m), ST(n, m);

   S=0; ST=0;
   if (n <= 2) throw mtl::logic_error("to small to coarse");

    dense_vector<Scalar>	 p(m), v(n);
    Vector			 c(n), ci(n), cit(n), f(n), u(n), t(n), r(n), di(n), vj(n); 
    c= 0; // no points in coarse grid
    f= 0; // no points in exclude of c
    u= 1; // all Points
    t= 0;
	//evaluation of S_i
	for (Size i = 0; i < m; i++) {
		for (Size k = 0; k < n; k++) {   //v= B[irange(0,imax)][i];  //not for commpressed matrix
			v[k]= B[k][i];
		}
		
		argmax= max_abs_pos(v);  
		for (Size j = 0; j < n ; j++ ) {
			if ((j != argmax) && (j != i) && abs(B[i][j]) >= beta*abs(v(argmax))) {
				S[i][j]=1;
			}
		}
	}

	//evaluation of S_i'  transposed  and p_i=#(S(i))
	for (Size i = 0; i < m; i++ ) {
		Size counter = 0;
		for (Size j = 0; j < n; j++) {
			if (S[i][j] == 1) {
				ST[j][i] = 1;
				counter++;
			}
		} 
		if (counter == 0) throw mtl::logic_error("isolateted point/node. Try to reduce coarse_beta.");
			
		p[i]= counter;
	}	

	while (sum(u) > 0) {
		Size i= max_abs_pos(p);
		if (i==ialt) {
			p[i]--;  // Stagnation in first Point
		}
		if (u(i) == 1) {
			c[i] = 1;
			u[i] = 0;
			v= ST.vector(i);
			for (Size j = 0; j < m;	j++) {
				//cut S_i' and u
				if (v[j] == 1 && u[j] == 1) {
					f[j] = 1;
					// cut S_j and u    p_k++
					vj= S.vector(j);
					for (Size l = 0; l < m; l++) {
						if (u[l] == 1 && vj[l] == 1) {
							p[l]++;
						}
					}
				}
			}
			//cut S_i and u   p_k--
			v= S.vector(i);
			for (Size j = 0; j < m; j++) {
				if (v[j] == 1 && u[j] == 1) {
					p[j]--;
				}
			}
			v= ST.vector(i);
			for (Size j = 0; j < m;	j++) {
				//cut S_i' and u
				if (v[j] == 1 && u[j] == 1) {
					u[j]=0;
				}
			}
		}
		ialt=i;
	}
	t=1;
	c=t-f;
	t=0;
	//part   ***Star
	r= f-t;
	while ( sum(r) > 0) {
		for ( Size i = 0; i < n; i++) {
			if (r[i] > 0) {
				t[i]=1;
				vj= S.vector(i);
				ci= c + vj;
				//ci =2 => 1
				for (Size k = 0; k < n; k++) {
					if (ci[k] > 1 ) {
						ci[k]= 1;
					} else {
						ci[k]= 0;
					}
				}
				di= vj - ci;
				cit = 0;   //forall j in D_i do
				for ( Size j = 0; j < n; j++) {
					if (di[j] == 1) {
						vj= S.vector(j);
						v= vj + ci;
						if ( max(v) == 1 ) {  //max(v)==1  <==> S_j cut C_i is empty
							if (max(cit) != 0) {
								c[i]= 1;
								f[i]= 0;
								goto loop;
							//TODO  GOTO part ****star**** ....done
							} else {
								cit[j]= 1;
								ci[j]=  1;
								di[j]=  0;
							}
						}
					}
				}
				//c union  c and cit  and f cut f and cit
				c= c + cit;
				f= f - cit;
				for (Size j = 0; j < n; j++) {
					if (c[j] > 1) {
						c[j]=1;
					}
					if (f[j] < 0) {
						f[j]= 0;
					}
				}	
			}
		loop:;  // GOTO marke ????????????? //
		}
	r=f-t;
	}
    return c;
}

///coarse_fine grid with notay algorithm
template < typename Matrix, typename Vector >
Vector inline amg_coarsepoints_notay(const Matrix& A, Vector& b, double beta)
{
   using std::abs;
   typedef typename mtl::Collection<Matrix>::value_type Scalar;
   typedef typename mtl::Collection<Matrix>::size_type  Size;
   Size n = num_cols(A);
   const Scalar    zero= math::zero(A[0][0]), one= math::one(A[0][0]);
	
   if (n <= 2) throw mtl::logic_error("to small to coarse");

    // algo 17.2  s351 Solving PDEs in C++ 

    // 1. init
    Vector c(n), f(n);
    c= 1;  // all points in coarse grid
    f= 0; // no points in exclude of c

    // 2. delete weak points
    for(Size i = 0; i < n; i++){
        if(c[i] == 1){
            for(Size j = 0; j < n; j++){
                if ( (j != i) && (abs(A[i][j]) >= beta*abs(A[i][i])) ){
                    c[j]= 0;
                    f[j]= 1;
                }
            }
        }
    }

    // 3. strong coupled points back to c
    for(Size i = 0; i < n; i++){
        if((c[i] == 0) && (A[i][i] != zero)){
            bool a=true;
            for(Size j = 0; j < n; j++){
                if(c[j] == 1){
                    if((A[i][j]/A[i][i]) < -beta){
                        a= false;
                        break;
                    }
                }
            }
            if(a){ // change c and f
	        c[i]= 1;
                f[i]= 0;
            }
        }
    }

    return c;
}

///omit the fine grid points
template < typename Matrix, typename Vector >
Matrix inline amg_restict_simple(const Matrix& A, Vector& c)
{	
        typedef typename mtl::Collection<Matrix>::size_type  Size;
	typedef typename mtl::Collection<Matrix>::value_type  Scalar;
	typedef typename mtl::Collection<Vector>::size_type vec_s_type;
	Size 		n= num_rows(A);
	vec_s_type  	counter=0;
	Matrix  R(counter,n);
	R=0;
	inserter<Matrix, update_plus<Scalar> > ins(R, counter);

	for (vec_s_type i = 0; i < n; i++) {
		if (c[i]) {
			counter++;
		}
	}
	
	
	counter=0;
	for (vec_s_type i = 0; i < n; i++) {
		if (c[i]) {
			ins[counter][i] << 1;
			counter++;
		}
	}
	return R;
}

///linear interpolation (average)
template < typename Matrix, typename Vector >
Matrix inline amg_restict_average(Matrix& A, Vector& c)
{	
        using std::abs;
   	typedef typename mtl::Collection<Matrix>::size_type   Size;
	typedef typename mtl::Collection<Matrix>::value_type  Scalar;
	typedef typename mtl::Collection<Vector>::size_type   vec_s_type;
	Size 		n= num_rows(A);
	vec_s_type  	counter(sum(c));
	Matrix 		R(counter,n);
	R=0;
	inserter<Matrix, update_plus<Scalar> > ins(R, counter);
		
	counter=0;
//	std::cout<< "c=" << c << std::endl;
	//std::cout<< "TEST__00" << std::endl;
	if (c[0]) {
		ins[0][0] << 1;
		ins[0][1] << 2;
		ins[0][2] << 1;
		counter++;
	}
	//std::cout<< "TEST__10" << std::endl;
	for (vec_s_type i = 1; i < n-2; i++) {
		if (c[i]) {
			ins[counter][i+1] << 2;
			ins[counter][i+2] << 1;
			ins[counter][i]   << 1;
			counter++;
		}
	}
// 	std::cout<< "TEST__20" << std::endl;
	if (c[n-2]) {
	//	std::cout<< "c_n-2" << std::endl;
		ins[counter][n-1] << 2;
		ins[counter][n-2] << 1;
		//R[counter][n-3]=1;
	}
// 	std::cout<< "TEST__30" << std::endl;
	if (c[n-1]) {
	//	std::cout<< "c_n-1" << std::endl;
		ins[counter][n-1] << 2;
		ins[counter][n-2] << 1;
	}
// 	std::cout<< "TEST__40" << std::endl;
	
	if (c[n-1] && c[n-2]) {
	//	std::cout<< "c_n-1 und c_n-2" << std::endl;
		ins[counter+1][n-1] << 0.5;
	}
// 	std::cout<< "TEST__40" << std::endl;

	//R*=0.25;
	//std::cout<< "R=\n" << R << "\n";
//	std::cout<< "R=\n" << trans(R) << "\n";
	return R;
}



///ALGO 17.4 Sovling PDEs in C++
template < typename Matrix, typename Vector >
Matrix inline amg_prolongate(Matrix& A, Vector& c, double beta)
{	
	//namespace tag= mtl::tag; using namespace mtl::traits;
	//using mtl::begin; using mtl::end;
        using std::abs;
   	using mtl::irange; using mtl::imax; 
   	typedef typename mtl::Collection<Matrix>::value_type Scalar;
   	typedef typename mtl::Collection<Matrix>::size_type  Size;
	typedef typename mtl::Collection<Vector>::value_type vec_v_type;
	typedef typename mtl::Collection<Vector>::size_type vec_s_type;

	const Scalar    zero= math::zero(A[0][0]), one= math::one(A[0][0]);//, beta=0.25;
	Size 		n = num_cols(A);
	vec_v_type	counter(sum(c));
	

	Vector f(c); f=1;
	// 1. init f and P, W
	f-= c;
	Matrix  P(A), W(A);
	W= zero;
	std::cout<< "test0" << std::endl;
	inserter<Matrix, update_plus<Scalar> > insp(P, counter);
	inserter<Matrix, update_plus<Scalar> > insw(W, counter);
	// 2. replace weak Points in P with zero
	for(Size i=0; i < n; i++){
		if( f[i] ){
			for(Size j=0; j < n; j++){
				if( (i!=j) && (P[i][i] != zero) && ((P[i][j]/P[i][i]) > -beta) ){
					insp[i][j] << zero;
				}
			}
		}
	}
	std::cout<< "test1" << std::endl;
	// 3. define B by P 
	Matrix B(P);

	// 4. replace i row for i in c
	irange all(0,imax);
	for(Size i = 0; i < n; i++){
		if(c[i]){
	//		P[irange(i,i+1)][all]= zero;
			insp[i][i] << one;
		}
	}
	std::cout<< "test2" << std::endl;
	// 5. define W[i][j]
	for(Size i=0; i < n; i++){
		for(Size j=0; j < n; j++){
			if((f[i]) && (f[j]) && (i!=j) && (P[i][j] != zero)){
				insw[i][j] << zero;
				for(Size k = 0; k < n; k++){
					if((c[k]) && (P[i][k] != zero)){
						insw[i][j] += B[j][k];
					}
				}
			}
		}
	}
	std::cout<< "test3" << std::endl;
	// 6. add fraction to P[i][]
	for(Size i=0; i < n; i++){
		for(Size j=0; j < n; j++){
			if(f[i] && f[j] && (i!=j) && (P[i][j] != zero)){
				for(Size k=0; k < n; k++){
					if(c[k] && (P[i][k] != zero) && (W[i][j] != zero)){
						insp[i][k]+= B[j][k]/W[i][j]*P[i][j];
					}
				}
			}
		}
	}
	
	// 7. drop column j for j in f
	counter=sum(c);
	//init ProlongateOperator
	Matrix Pro(n,counter);
	Pro= zero;
	inserter<Matrix, update_plus<Scalar> > inspro(Pro, counter);
	
	int lastCol=0;
	for(Size j=0; j < n; j++){
		if(c[j]){
			//Pro[irange(0,imax)][irange(lastCol,lastCol+1)] == P[irange(0,imax)][irange(j,j+1)];
			for ( Size i = 0; i < n; i++) {
				inspro[i][lastCol] << P[i][j];
			}
			lastCol++;
		}
	}
	Matrix Pro_norm(Pro);
	inserter<Matrix, update_times<Scalar> > inspro2(Pro_norm, counter);
	std::cout<< "test5" << std::endl;
	// 8. normalizise rows of Prolongation
	dense_vector<Scalar> row_sum(n);
	for(Size i=0;i<n;i++){
		row_sum[i]= zero;
		for(Size j=0;j<counter; j++){
			row_sum[i]+= Pro[i][j];
		}
	}
	std::cout<< "test6" << std::endl;
	for(Size i = 0; i < n; i++){
		if (row_sum[i] != zero){
			//Pro[irange(i,i+1)][irange(0,imax)]=Pro[irange(i,i+1)][irange(0,imax)]/row_sum[i];
			for (Size j = 0; j < n; j++) {
				inspro2[i][j] << Pro[i][j]/row_sum[i];
			}
		}
	}
	std::cout<< "test7" << std::endl;
	//Matrix Q(counter,counter);
	//Q= trans(Pro)*A*Pro;
	
	return Pro_norm;
}


/// P Prolongation and R restrikction for f_smoother
template < typename Matrix, typename Vector >
std::pair<Matrix, Matrix >
inline amg_prolongate_f(const Matrix& A, Vector& b, Vector& c)
{

	
   	using mtl::irange; using mtl::imax;
   	typedef typename mtl::Collection<Matrix>::value_type Scalar;
   	typedef typename mtl::Collection<Matrix>::size_type  Size;

	Size 		n = num_cols(A), k(sum(c));
	Vector 		f(n),b_permut(b);
	Matrix 		Permutation(n,n);

	

	//F-smoother------------------------------
		std::cout<< "c=" << c << "\n";
		std::cout<< "counter=k=" << k << "\n";
		//Permutation of Finegridpoints upwardly
		for (Size i = 0; i < n; i++) {
			f[i]=i;
		}
		for (Size j = 0; j < n-1; j++) {
			for (Size i = j; i < n-1; i++) {
				if (c(i) == 0) {
					c(i)=c(i+1); c(i+1)=0;
					int tmp= f(i);
					f(i)=f(i+1); f(i+1)=tmp;
				}
			}
		}
		
		std::cout<< "Permutationsvector=" << f << "\n";
		Permutation= permutation(f);
   		std::cout << "\nP =\n" << Permutation;
		Matrix A_l(trans(Permutation)*A*Permutation);
		
		
		b_permut= Permutation*b;
		std::cout<< "b_permut=" << b_permut << "\n";
   		std::cout << "\nA_l =\n" << A_l;
	
		Matrix A_FF(k,k), A_FC(k, n-k), A_CF(n-k, k), A_CC(n-k, n-k), S(n,n), AFC(n,n);
		A_FF= A_l[irange(0,k)][irange(0,k)];
		A_FC= A_l[irange(0,k)][irange(k,imax)];
		A_CF= A_l[irange(k,imax)][irange(0,k)];
		A_CC= A_l[irange(k,imax)][irange(k,imax)];
		AFC[irange(0,k)][irange(0,k)]=       A_FF;
		AFC[irange(0,k)][irange(k,imax)]=    A_FC;
		AFC[irange(k,imax)][irange(0,k)]=    A_CF;
		AFC[irange(k,imax)][irange(k,imax)]= A_CC;		
		
		Matrix Identity(n-k,n-k), P(n,k), R(k,n);
		Identity= 1; P=0;
		P[irange(0,k)][irange(0,n-k)]= -1*inv(A_FF)*A_FC;
		P[irange(k,imax)][irange(0,imax)]= Identity;
		//std::cout<< "P=" << P << "\n";
		R[irange(0,imax)][irange(0,n-k)]= -1*A_CF*inv(A_FF);
		R[irange(0,imax)][irange(n-k,imax)]=Identity;
		//std::cout<< "R=" << R << "\n";


	return std::make_pair(P,R);
}



///only S  
template < typename Matrix, typename Vector >
Matrix inline amg_f_smoother(const Matrix& A, Vector& c)
{	
        using std::abs;
   	using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper;
   	typedef typename mtl::Collection<Matrix>::value_type Scalar;
   	typedef typename mtl::Collection<Matrix>::size_type  Size;
	
	Size 		n = num_cols(A), k(sum(c));
	
	//F-smoother------------------------------
	
	Matrix A_FF(k,k), S(n,n);
	A_FF= A[irange(0,k)][irange(0,k)];
	S=0;
	S[irange(0,k)][irange(0,k)]= inv(A_FF);
	//std::cout<< "S=\n" << S << "\n";
	
	return S;
}

///Return %matrix A=[A_FF A_FC; A_CF AFF] and %vector b=[b_ff, b_cc] A_ff \in R^(k x k) and k=sum(c)
template < typename Matrix, typename Vector >
Matrix inline amg_fine_coars_permutation(const Matrix& A, Vector& b, Vector& c)
{	
        using std::abs;
   	using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper;
   	typedef typename mtl::Collection<Matrix>::value_type Scalar;
   	typedef typename mtl::Collection<Matrix>::size_type  Size;
	typedef typename mtl::Collection<Vector>::value_type vec_v_type;
	typedef typename mtl::Collection<Vector>::size_type vec_s_type;

	Size 		n = num_cols(A), k(sum(c));
	Vector 		f(n),b_p(b);
	Matrix 		Permutation(n,n);

	
	for (Size i = 0; i < n; i++) {
		f[i]=i;
	}
	for (Size j = 0; j < n-1; j++) {
		for (Size i = j; i < n-1; i++) {
			if (c(i) == 0) {
				c(i)=c(i+1); c(i+1)=0;
				int tmp= f(i);
				f(i)=f(i+1); f(i+1)=tmp;
			}
		}
	}
	//R= reorder(f);
	std::cout<< "Permutationsvector=" << f << "\n";
	Permutation= permutation(f);
	std::cout << "\nP =\n" << Permutation;
	Matrix A_p(trans(Permutation)*A*Permutation);
	b_p= Permutation*b;
		
	return std::make_pair(A_p, b_p);
}






}} // namespace mtl::matrix

#endif // MTL_MATRIX_AMG_OP_INCLUDE
