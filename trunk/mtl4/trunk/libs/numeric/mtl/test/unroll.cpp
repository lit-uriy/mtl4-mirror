// $COPYRIGHT$

#include <iostream>
#include <vector>
#include <boost/test/minimal.hpp>
#include <boost/timer.hpp>

using namespace std;

int const vector_size = 1000; // 1000

vector<double> gv1(vector_size, 2.0), gv2(vector_size, 3.0);

template <typename F>
void time_dot(std::string fname, F f)
{

    boost::timer start;
    double result;
    for (int i= 0; i < 100000; i++) // 100000
	result= f(gv1, gv2);
    double duration = start.elapsed();
    cout << fname << ": " << duration << "s, result = " << result << "\n";
}

double dot(vector<double> const& v1, vector<double> const& v2)
{
    double sum= 0.0;
    for (unsigned i= 0; i < v1.size(); i++)
	sum+= v1[i] * v2[i];

    return sum;
}

double dot2(vector<double> const& v1, vector<double> const& v2)
{
    double sum= 0.0, sum2 = 0.0;
    for (unsigned i= 0; i < v1.size(); i+= 2) {
	sum+= v1[i] * v2[i];
	sum2+= v1[i+1] * v2[i+1];
    }

    return sum + sum2;
}

double dot4(vector<double> const& v1, vector<double> const& v2)
{
    double sum= 0.0, sum2 = 0.0, sum3= 0.0, sum4= 0.0;
    for (unsigned i= 0; i < v1.size(); i+= 4) {
	sum+= v1[i] * v2[i];
	sum2+= v1[i+1] * v2[i+1];
	sum3+= v1[i+2] * v2[i+2];
	sum4+= v1[i+3] * v2[i+3];
    }

    return sum + sum2 + sum3 + sum4;
}


template <unsigned Depth>
struct recursive_data
{
    static unsigned const    depth= Depth;
    double                   sum;
    recursive_data<Depth-1>  remainder;

    recursive_data(double s) : sum(s), remainder(s) {}
    
    double sum_up()
    {
	// cout << "l" << depth << ' ' << sum << ", ";
	return sum + remainder.sum_up();
    }
};
       
template <>
struct recursive_data<1> 
{
    static unsigned const    depth= 1;
    double                   sum;

    recursive_data(double s) : sum(s) {}
    
    double sum_up()
    {
	// cout << "l1 " << sum << '\n';
	return sum;
    }
};

template <unsigned Depth, unsigned MaxDepth>
struct dot_block
{
    static unsigned const offset= MaxDepth - Depth;
    double                   sum;
    dot_block<Depth-1, MaxDepth> remainder;

    dot_block() : sum(0.0), remainder() {}

    void operator() (vector<double> const& v1, vector<double> const& v2, 
		     unsigned i, double sum[MaxDepth])
    {
	sum[offset]+= v1[ i + offset ] * v2[ i + offset ];
	remainder (v1, v2, i, sum);
    }
    double sum_up()
    {
	// cout << "l" << Depth << ' ' << sum << ", ";
	return sum + remainder.sum_up();
    }
};

template <unsigned MaxDepth>
struct dot_block<1, MaxDepth>
{
    static unsigned const offset= MaxDepth - 1;
    double                   sum;
    dot_block() : sum(0.0) {}

    void operator() (vector<double> const& v1, vector<double> const& v2, 
		     unsigned i, double sum[MaxDepth])
    {
	sum[offset]+= v1[ i + offset ] * v2[ i + offset ];
    }
    double sum_up()
    {
	// cout << "l1 " << sum << '\n';
	return sum;
    }
};

template <unsigned Depth>
double unrolled_dot(vector<double> const& v1, vector<double> const& v2)
{
    // check v1.size() == v2.size();
    unsigned size= v1.size(), blocks= size / Depth, blocked_size= blocks * Depth;
    double sum[Depth];
    for (unsigned i= 0; i < Depth; i++) sum[i]= 0.0;



    // recursive_data<Depth> sum_block(0.0);

    dot_block<Depth, Depth> dot_object;
    for (unsigned i= 0; i < blocked_size; i+= Depth)
	dot_object(v1, v2, i, sum);

    // double sum= dot_object.sum_up();
    for (unsigned i= blocked_size; i < size; ++i)
	sum[0]+= v1[i] * v2[i];
    for (unsigned i= 1; i < Depth; i++) sum[0]+= sum[i];

    return sum[0];
}


int test_main(int argc, char* argv[])
{

    time_dot("regular   ", dot);
    time_dot("unrolled 2", dot2);
    time_dot("unrolled 4", dot4);
    
    cout << "--------------------\n";
    time_dot("unrolled 2", unrolled_dot<2>);
    time_dot("unrolled 3", unrolled_dot<3>);
    time_dot("unrolled 4", unrolled_dot<4>);
    time_dot("unrolled 5", unrolled_dot<5>);
    time_dot("unrolled 6", unrolled_dot<6>);
    time_dot("unrolled 7", unrolled_dot<7>);
    time_dot("unrolled 8", unrolled_dot<8>);
    time_dot("unrolled 9", unrolled_dot<9>);
    time_dot("unrolled10", unrolled_dot<10>);
    time_dot("unrolled12", unrolled_dot<12>);
    time_dot("unrolled14", unrolled_dot<14>);
    time_dot("unrolled16", unrolled_dot<16>);

    return 0;
}
