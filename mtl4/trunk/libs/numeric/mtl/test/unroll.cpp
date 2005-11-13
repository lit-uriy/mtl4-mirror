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
	return sum;
    }
};

template <unsigned Depth, unsigned MaxDepth>
struct dot_block
{
    inline void operator() (vector<double> const& v1, vector<double> const& v2,
		     recursive_data<Depth>& sum_block, unsigned i)
    {
	static unsigned const depth= sum_block.depth;
	sum_block.sum+= v1[ i + MaxDepth - depth ] * v2[ i + MaxDepth - depth ];
	// dot_block<recursive_data<depth-1>, MaxDepth>(v1, v2, sum_block.remainder, i);
	dot_block<depth-1, MaxDepth>() (v1, v2, sum_block.remainder, i);
    }
};

template <unsigned MaxDepth>
struct dot_block<1, MaxDepth>
{
    // static const unsigned offset= MaxDepth - 1;

    inline void operator() (vector<double> const& v1, vector<double> const& v2,
		     recursive_data<1>& sum_block, unsigned i)
    {
	sum_block.sum+= v1[ i + MaxDepth - 1 ] * v2[ i + MaxDepth - 1 ];
    }
};

template <unsigned Depth>
double unrolled_dot(vector<double> const& v1, vector<double> const& v2)
{
    // check v1.size() == v2.size();
    unsigned size= v1.size(), blocks= size / Depth, blocked_size= blocks * Depth;

    recursive_data<Depth> sum_block(0.0);
    for (unsigned i= 0; i < blocked_size; i+= Depth)
	dot_block<Depth, Depth>()(v1, v2, sum_block, i);

    double sum= sum_block.sum_up();
    for (unsigned i= blocked_size; i < size; ++i)
	sum+= v1[i] * v2[i];
    return sum;
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
