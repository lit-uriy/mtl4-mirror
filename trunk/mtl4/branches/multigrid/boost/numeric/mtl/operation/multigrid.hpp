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

// With contributions from Cornelius Steinhardt

#ifndef MTL_MATRIX_MULTIGRID_INCLUDE
#define MTL_MATRIX_MULTIGRID_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/householder.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/sum.hpp>
#include <boost/numeric/itl/smoother/gauss_seidel.hpp>
#include <boost/numeric/itl/krylov/cg.hpp>
#include <time.h>
#include <string>


namespace mtl { namespace matrix {


template <typename Matrix, typename Vector>
struct MGLEVEL
{
	Matrix A;
	Vector x;
	Vector b;//rhs
	Matrix P;//prolongation operator
	Matrix R;//restiction operator
	Matrix S_pre;  //pre-smooth operator
	Matrix S_post; //post-smooth operator
};




template < typename Matrix, typename Vector, typename Iteration>
int inline multigrid_algo(const Matrix& A, Vector& x, Vector& b, Iteration& iter, int maxLevel, char* coarse, char *prolongate, char *restriction, int ny, int ny_pre, int ny_post, char *pre_smooth, char *post_smooth, double omega, double coarse_beta, char *coarse_op) //setup
{
	Vector r(b-A*x);
	long int old_time = 0, new_time = 0;
	//std::cout<< "------------START------multigrid_algo--------" << std::endl;
	old_time= clock();
	std::vector<MGLEVEL<Matrix, Vector> > list(multigrid_setup(A, x, b, maxLevel, coarse, prolongate, restriction, pre_smooth, post_smooth, omega, coarse_beta, coarse_op)); //init all MG levels
	new_time= clock();
	
	std::cout<< "setup finished in " << (float)(new_time-old_time)/CLOCKS_PER_SEC << "sec.\n";
	//v oder w cycle extra funktionen
	//for (int i=0; i < list.size(); i++) {
	//	std::cout<< "A[" << i << "]=\n" << list[i].A << std::endl;
	//}
	
	while (! iter.finished(r)) {
		old_time = 0, new_time = 0;
		old_time= clock();
		list[0].x= x;
		if (ny == 1) {
	//	std::cout<< "v-cycle" << std::endl;
			x= multigrid_vw_cycle(list, 0, 1, ny_pre, ny_post, coarse_op); 
		} else if (ny > 1) {
	//	std::cout<< "w-cycle" << std::endl;
			x= multigrid_vw_cycle(list, 0, ny, ny_pre, ny_post, coarse_op);
		} else { //same as w-cycle  just without smoothing
	//	std::cout<< "full-cycle" << std::endl;
			x= multigrid_full_cycle(list, 0, ny, ny_pre, ny_post, coarse_op);
		}
		r=b-A*x;
		//std::cout<< "two_norm(r)=" << two_norm(r)<< std::endl;
		++iter;
		new_time= clock();
		//std::cout<< "time while " << (float)(new_time-old_time)/CLOCKS_PER_SEC << "sec.\n";
		
	}


	return iter.error_code();
}

///v-cycle and w-cycle
template < typename Matrix, typename Vector >
Vector inline multigrid_vw_cycle(std::vector<MGLEVEL<Matrix, Vector> >& list, int level, int ny, int ny_pre, int ny_post, char *coarse_op)
{
	typedef typename mtl::Collection<Matrix>::size_type  Size;
	typedef typename mtl::Collection<Matrix>::value_type Scalar;
	Size maxLevel(list.size());
	//std::cout<< "START-----vw_cycle on level=" << level << "___ maxLevel=" << maxLevel << std::endl;

	MGLEVEL<Matrix, Vector>&      mg= list[level];
	Vector&       x= mg.x;
	const Vector& b= mg.b;
	Matrix&       A= mg.A; 
        itl::gauss_seidel<Matrix, Vector> gs(A, b);	
	
	//v_cycle algo
	if (level == maxLevel-1) {
		x= A * b;	
	} else {
		//pre smoothing
		for (int i = 0; i < ny_pre; i++) 
			gs(x);  //x+= mg.S_pre * (b - A*x);	
	
		Vector  r(b - A*x);
		list[level+1].b= mg.R * r;

		//recursion of multigrid method
		for (int i = 0; i < ny; i++)
			x+= 0.5 * mg.P * multigrid_vw_cycle(list, level+1, ny, ny_pre, ny_post, coarse_op);

		//post smoothing
		for (int i = 0; i < ny_post; i++) 			
			gs(x);   //x+=mg.S_post * (b - A * x); 
	}
	
	//std::cout<< "END-------vw_cycle on level=" << level << "\n";
	return x;
}

///full multigrid cycle without smoothing
template < typename Matrix, typename Vector >
Vector inline multigrid_full_cycle(std::vector<MGLEVEL<Matrix, Vector> > list, int level, int ny, int ny_pre, int ny_post, char *coarse_op)
{
	typedef typename mtl::Collection<Matrix>::size_type  Size;
	Size maxLevel(list.size());
	std::cout<< "START-----full_cycle on level=" << level << "___ maxLevel=" << maxLevel << std::endl;
	
	MGLEVEL<Matrix, Vector>& mg= mg;
	//full_cycle algo
	if (level == maxLevel-1) {
		mg.x= mg.A*mg.b;
	} else {
		Vector  r(mg.b-mg.A*mg.x);
		list[level+1].b= mg.R*r;

		//recursion of full multigrid method
		Vector enew;
		if (ny > 0) { //ny==1 -> v-cycle   ny > 1  -> w-cycle with ny_level
			for (int i = 0; i < ny; i++) {
				enew= multigrid_full_cycle(list, level+1, ny, ny_pre, ny_post, coarse_op);
			}
		} else {
			throw mtl::logic_error("ny needs to be greater than 1");
		}
		mg.x= mg.x + mg.P*enew;
	}
	return mg.x;
}




///multigrid setup phase. Init restiction and prolongation operator and %matix on all level
template < typename Matrix, typename Vector >
std::vector<MGLEVEL<Matrix, Vector> > inline multigrid_setup(const Matrix& A, Vector& x, Vector& b, int maxLevel, char *coarse, char *prolongate, char *restriction,
char *pre_smooth, char *post_smooth, double omega, double coarse_beta, char *coarse_op)   ////setup
{

	typedef typename mtl::Collection<Matrix>::size_type  Size;

	std::vector<MGLEVEL<Matrix, Vector> > MGLEVEL_LISTE(maxLevel);
	Size  k;

	MGLEVEL_LISTE[0].A= A;
	MGLEVEL_LISTE[0].x= x;
	MGLEVEL_LISTE[0].b= b;
	Matrix 		A_i(A);
	Vector		b_i(b), x_i(x);

	for (int i = 0; i < maxLevel-1; i++) {
		std::cout<< "Setup i=" << i << std::endl;
		//coarse %matrix A
		dense_vector<int> c(num_cols(A_i));
		if (coarse == "rs"){
// 			std::cout<< "start coarse rs" << std::endl;
			c= amg_coarsepoints_rs(A_i, c, coarse_beta);
// 			std::cout<< "end coarse rs" << std::endl;
		} else if (coarse == "notay") {
// 			std::cout<< "start coarse notay" << std::endl;
			c= amg_coarsepoints_notay(A_i, c, coarse_beta);
// 			std::cout<< "end coarse notay" << std::endl;
		} else if (coarse == "simple") {
// 			std::cout<< "start coarse simple" << std::endl;
			c= amg_coarsepoints_simple(A_i, c);
// 			std::cout<< "end coarse simple" << std::endl;
		} else if (coarse == "default") {
// 			std::cout<< "start coarse default" << std::endl;
			c= amg_coarsepoints_default(A_i, c);
// 			std::cout<< "end coarse default" << std::endl;
		} else {
			std::cout<< "Wrong coarsening method. Try: rs, notay, simple, default." << std::endl;
		}

		k= sum(c);
		if ( k == size(c) ) {
			std::cout<< "stagnation while coarsening.. try other methode or other beta." << std::endl;
			std::cout<< "The maximum level is reduced to "<< i+1 << std::endl;
			maxLevel = i+1;
			MGLEVEL_LISTE.resize(i+1);
			if ( coarse_op == "inv") {
				MGLEVEL_LISTE[i].A= inv(A_i);
			} else if (coarse_op == "cg") { 
				MGLEVEL_LISTE[i].A= A_i;
			}
			break;
		}
// 		std::cout<< "c=" << c << "\n";
// 		std::cout<< "dim=" << k << std::endl;
		//prolongation operator P and restriction operator R
		Matrix		R(k, num_cols(A_i)), P(num_rows(A_i), k);

		if (prolongate == "trans" && restriction == "trans") {
			std::cout<< "Select at least one methode of restiction or prolongation" << std::endl;
			break;
		} 
	//	std::cout<< "runtimtest 1" << std::endl;		
		//restiction operator R				///TODO   Randbedingungen???
		if (restriction == "simple") {
			R=amg_restict_simple(A_i,c);
		} else if (restriction == "avg") {
// 			std::cout<< "start restrict avg" << std::endl;
			R=amg_restict_average(A_i, c);
// 			std::cout<< "end restrict avg" << std::endl;
	//	} else if (restriction == "trans") {
	//		R=trans(P);
		} else {
			std::cout<< "Wrong restriction method. Try: simple or avg." << std::endl;
			break;
		}
	//	std::cout<< "runtimtest 2" << std::endl;
		//prolongation operator P (interpolation)
		if (prolongate == "trans") {
// 			std::cout<< "start prolongate trans" << std::endl;
			P= trans(R);
// 			std::cout<< "end prolongate trans" << std::endl;
		} else if (prolongate == "normal") {
			P=amg_prolongate(A_i, c, coarse_beta);
		} else {
			std::cout<< "Wrong prolongation method. Try: trans or normal." << std::endl;
			break;
		}
		R= trans(P);
		//std::cout<< "runtimtest 3" << std::endl;
		MGLEVEL_LISTE[i].P= P;
		MGLEVEL_LISTE[i].R= R;
		//std::cout<< "P=\n" << P << "\n";
		//construct new Level
		Vector rnew(k), nullvector(k), r(MGLEVEL_LISTE[i].b-MGLEVEL_LISTE[i].A*MGLEVEL_LISTE[i].x);
		nullvector= 0;	rnew=0;
		//rnew= R*r;
		A_i.change_dim(k, k);
		A_i=(R * MGLEVEL_LISTE[i].A * P);
		MGLEVEL_LISTE[i+1].A= A_i;
		MGLEVEL_LISTE[i+1].x= nullvector;
		MGLEVEL_LISTE[i+1].b= rnew;
		//std::cout<< "runtimtest 4" << std::endl;
		if (i+1 == maxLevel-1) {
			MGLEVEL_LISTE[i+1].A= inv(A_i);//??? in last level inv(A)??????????????
		}

		//pre_smoother on level i
		if (pre_smooth == "gauss_seidel") {
			Matrix	D(MGLEVEL_LISTE[i].A - strict_lower(MGLEVEL_LISTE[i].A) - strict_upper(MGLEVEL_LISTE[i].A));
			invert_diagonal(D);
			MGLEVEL_LISTE[i].S_pre= D;
		} else if (pre_smooth == "jacobi") {
			Matrix	D(MGLEVEL_LISTE[i].A - strict_lower(MGLEVEL_LISTE[i].A) - strict_upper(MGLEVEL_LISTE[i].A));
			invert_diagonal(D);
			MGLEVEL_LISTE[i].S_pre= omega*D;
		}  ///TODO  red black smoother


		//post_smoother on level i
		if (post_smooth == "gauss_seidel") {
			Matrix	D(MGLEVEL_LISTE[i].A - strict_lower(MGLEVEL_LISTE[i].A) - strict_upper(MGLEVEL_LISTE[i].A));
			invert_diagonal(D);
			MGLEVEL_LISTE[i].S_post= D;
		} else if (post_smooth == "jacobi") {
			Matrix	D(MGLEVEL_LISTE[i].A - strict_lower(MGLEVEL_LISTE[i].A) - strict_upper(MGLEVEL_LISTE[i].A));
			invert_diagonal(D);
			MGLEVEL_LISTE[i].S_post= omega*D;
		}///TODO  red black smoother
	}

	
// 	MGLEVEL_LISTE[0].A= A;
// 	std::cout<< "MGLEVEL.A"<< MGLEVEL_LISTE[0].A << std::endl;
	return MGLEVEL_LISTE;
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_MULTIGRID_INCLUDE
