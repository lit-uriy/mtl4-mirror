#include <iostream>

#include <boost/numeric/linear_algebra/operators.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/concepts.hpp>

#include "algebraic_functions.hpp"

using math::identity; using math::add;

using std::string;

namespace math 
{
    template<> 
    struct identity< math::add<string>, string >  
    { 
        string operator()(const string&) const 
        {
    	return string(); 
        } 
    } ;

# ifdef LA_WITH_CONCEPTS
    concept_map Monoid< math::add<string>, string > {}
# endif // LA_WITH_CONCEPTS
}        

int main(int, char* []) 
{
    using namespace std;
    using namespace mtl;

    string               sa("a"), sab("ab"), sbc("bc"), sc("c"), s;
    math::add<string>    string_add;

    cout << "equal_results(sa, sbc,  sab, sc, string_add) " 
	 << equal_results(sa, sbc,  sab, sc, string_add)  << endl;
    cout << "equal_results(sab, sbc,  sab, sc, string_add) " 
	 << equal_results(sab, sbc,  sab, sc, string_add) << endl;
    
    cout << "identity_pair(s, s, string_add) " 
	 << identity_pair(s, s, string_add)  << endl;
    cout << "identity_pair(sa, sa, string_add) " 
	 << identity_pair(sa, sa, string_add)  << endl;

    cout << "multiply_and_square(sab, 13, string_add) "
	 << multiply_and_square(sab, 13, string_add) << endl;
    return 0;
}
