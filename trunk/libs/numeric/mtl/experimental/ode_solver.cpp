#include <iostream>
#include <fstream>
#include <utility>
#include "time.h"
#include <sstream>

#include <boost/timer.hpp>
#include <boost/numeric/mtl/mtl.hpp>


typedef double value_type;
typedef std::size_t size_type;
typedef mtl::dense_vector<value_type>  vector_type;
typedef mtl::compressed2D<value_type>  matrix_type;
 const double pi(3.14159265358979323846);

template <typename Vector, typename Matrix>
class grad_f_ftor
{
  public:
    grad_f_ftor(const Vector& u,  const Matrix& M, const Matrix& K, const Matrix& G, const double& beta, const double& w) 
      : u(u), M(M), K(K), G(G), beta(beta), w(w)
      {
	q.change_dim(num_cols(G));
	q=0.0;
	x.change_dim(num_cols(G));
	x=0.0;
	for(size_type i=1; i< size(x); i++)
	  x(i)=x(i-1)+0.01;
	std::cout<< "x=" << x << "\n";
	
	size_type half(num_cols(G)/2);
	value_type  value(0.4);
	for(size_type i= 0; i < 4; i++){   //TODO   fahrendes q  bzw  variables.
	    q(half-i)= value;
	    q(half+i)= value;
	    value-=0.1;
	}
      }
      
    //get timedependent input q for the heating
    Vector get_input(const double& time) 
    {
      value_type center(0.5*sin(2*pi*time)+0.5);
      std::cout<< "center="<< center << "\n";
    
      for(size_type i=0; i< size(x); i++)
	q(i)=beta*cos(std::min(std::abs(pi*(x(i)-center)/(2*w)),pi/2));
      //q=beta*cos(std::min(std::abs(pi*(x-c)/(2*w)),pi/2));
      return q;
    }

    Vector operator()(const Vector& u, const double& time) 
    {
	q=get_input(time);
	Vector x(K*u + G*q);
	return x;
    }
 
  private:
    Vector      u, q, x;
    Matrix      M,K,G;
    double      beta, w;
   
};



template <typename grad_f_ftor, typename Vector, typename Matrix>
Vector ode23s(grad_f_ftor func, value_type start_time, value_type end_time, Vector start_value, Matrix M, Matrix K){

  value_type time(start_time), time_step(0.005);
  value_type  h= 0.005, gamma(1-1/sqrt(2));//, gamma_21(-gamma/0.5);
  mtl::dense2D<value_type> A(5,5);
  A=1;
  mtl::dense2D<value_type> LU(M-h*gamma*K);
//  std::cout<< "start LU von " << num_rows(LU) << " x "<< num_cols(LU) <<"\n";
  boost::timer lu_timer;
  lu(LU);
  Vector x(start_value);
//  std::cout<< " lu tok  " << lu_timer.elapsed() << "\n";
 
   while (time < end_time){
//      std::cout<< "time=" << time << "\n";
//      boost::timer lu_solve_timer;
      Vector k1( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), func(x,time))) ); //FIXME  time dependent
 //     std::cout<< " lu tok  " << lu_solve_timer.elapsed() << "\n";
    
      Vector step1(x+h*k1);
//      Vector step2(func(step1,time+time_step)+h*gamma_21*K*k1);
      Vector step2(func(step1,time+time_step)-2*k1);
      Vector k2( upper_trisolve(upper(LU), unit_lower_trisolve(strict_lower(LU), step2)) );
      
      x+= 3/2*h*k1+1/2*h*k2;
//      std::cout<< "norm_k1=" << two_norm(k1) << " norm_k2=" << two_norm(k2) << "\n";
      time+= time_step;
  }
  return x;
}

//save current state of solution x(time)
template<typename Vector>
void save_data(const Vector& x, const double& time) 
{
    std::string name;

    name << "plot_data_" << time <<".dat";
    std::cout << name.str() << "\n";;


    std::fstream f;
    f.open(name.c_str(), std::ios::out);
    size_type n(1111), n1(101);
    

    size_type row(0), col(0);
    for(size_type i=0; i < n; i++){
      if(i%(n/n1)==0){
	row=0;col+=1;
	f << "\n";
      }
      f <<  col << " " << row << " " << x(i) << "\n";
      row+=1;
    }
    f.close();
    
}



 
int main( int  , char ** )
{
     //read jörgs matrices
    matrix_type M(mtl::io::matrix_market( "M.mtx"));
    matrix_type K(mtl::io::matrix_market( "K.mtx"));
    matrix_type G(mtl::io::matrix_market( "G.mtx"));
    
    size_type n(num_rows(K));
    value_type  start_time(0.0),
		end_time(0.1);
    vector_type  x(n,0.0), x0(n,0.0), u(n,0.0);
        
    grad_f_ftor< vector_type, matrix_type >  grad_f(u, M, K, G, 0.2, 0.1);
    
    x= ode23s(grad_f, start_time, end_time, x0, M, K);
    
    save_data(x, end_time);
    
    std::cout<< "finish\n";

    return 0;

}