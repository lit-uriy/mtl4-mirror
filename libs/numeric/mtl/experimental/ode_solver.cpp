#include <iostream>
#include <fstream>
#include <utility>
#include "time.h"

#include <boost/timer.hpp>
#include <boost/numeric/mtl/mtl.hpp>


typedef double value_type;
typedef std::size_t size_type;
typedef mtl::dense_vector<value_type>  vector_type;
typedef mtl::compressed2D<value_type>  matrix_type;


template <typename Vector, typename Matrix>
class grad_f_ftor
{
  public:
    grad_f_ftor(const Vector& u,  const Matrix& M, const Matrix& K, const Matrix& G) 
      : u(u), M(M), K(K), G(G)
      {
	q.change_dim(num_cols(G));
	q=0.0;
	size_type half(num_cols(G)/2);
	value_type  value(0.4);
	for(size_type i= 0; i < 4; i++){   //TODO   fahrendes q  bzw  variables.
	    q(half-i)= value;
	    q(half+i)= value;
	    value-=0.1;
	}
      }

    Vector operator()(const Vector& u) const
    {
	Vector x(K*u + G*q);
	return x;
    }
 
  private:
    Vector      u, q;
    Matrix      M,K,G;
};



template <typename grad_f_ftor, typename Vector, typename Matrix>
Vector ode23s(grad_f_ftor func, value_type start_time, value_type end_time, Vector start_value, Matrix M, Matrix K){

  value_type time(start_time), time_step(0.05);
  value_type  h= 0.01, gamma(1+1/sqrt(2)), gamma_21(0.5);
  mtl::dense2D<value_type> A(5,5);
  A=1;
  mtl::dense2D<value_type> LU(M-h*gamma*K);
  std::cout<< "start LU von " << num_rows(LU) << " x "<< num_cols(LU) <<"\n";
  boost::timer lu_timer;
  lu(LU);
   Vector x(start_value);
   std::cout<< " lu tok  " << lu_timer.elapsed() << "\n";
 
   while (time < end_time){
     std::cout<< "time=" << time << "\n";
   boost::timer lu_solve_timer;
 // Vector k1( lu_solve(LU_old, func(x)) ); //FIXME  time dependent
  Vector k1( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), func(x))) );
   std::cout<< " lu tok  " << lu_solve_timer.elapsed() << "\n";
 
  //std::cout<< "k1=" << k1 << "\n";
 // Vector test(func(x)-LU_old*k1);
 // std::cout<< "test=" << test << "\n  and norm="<< two_norm(test) << "\n";
 // Vector v2( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), func(x))) );
  //Vector new_test(v2-k1);
  //std::cout<< "norm_new=" << two_norm(new_test) << "\n";
  
  Vector step1(x+h*k1);
  Vector step2(func(step1)+h*gamma_21*K*k1);
//  Vector k2( lu_solve(LU, step2) );
  Vector k2( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), step2)) );
  
//  Vector test1(func(step1)+h*gamma_21*K*k1-LU*k2);
//  std::cout<< "test1=" << test1 << "\n  and norm="<< two_norm(test1) << "\n";
  x+= 3/2*h*k1+1/2*h*k2;
  std::cout<< "norm_k1=" << two_norm(k1) << " norm_k2=" << two_norm(k2) << "\n";
  time+= time_step;
  }
  return x;
}



 
int main( int argc , char **argv )
{
     //read jörgs matrices
    matrix_type M(mtl::io::matrix_market( "M.mtx"));
    matrix_type K(mtl::io::matrix_market( "K.mtx"));
    matrix_type G(mtl::io::matrix_market( "G.mtx"));
    
    std::cout<< "G=\n" << num_rows(G) <<" , " << num_cols(G) << "\n";
  
  
    std::cout << "Testing ros2 method \n";
    size_type n(num_rows(K)), n1(num_cols(G));
    vector_type  x(n,0.0), x0(n,0.0), u(n,0.0);
        
    grad_f_ftor< vector_type, matrix_type >  grad_f(u, M, K, G);
    std::cout<< "grad_f(u)=" << grad_f(u) << "\n";
    
    
 #if 1   
    x= ode23s(grad_f, 0.0, 1.0, x0, M, K);
    std::cout<< "x=" << x << "\n";
#endif
    std::fstream f;
    f.open("plot_data.dat", std::ios::out);
   
    

    size_type row(0), col(0);
    for(size_type i=0; i < n; i++){
      if(i%(n/n1)==0){
	row=0;col+=1;
	f << "\n";
      }
      f <<  row << " " << col << " " << x(i) << "\n";
      row+=1;
    }
    f.close();
    
    std::cout<< "finish\n";
    return 0;

}