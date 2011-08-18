// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <boost/type_traits/remove_const.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/is_bitwise_serializable.hpp>
#include <boost/mpl/assert.hpp>
#include <iostream>


using namespace std;  
    
template <typename T>
void test(T, const char* name)
{
    using namespace boost;
    cout << "Test " << name << '\n';
    BOOST_MPL_ASSERT((serialization::is_bitwise_serializable<typename remove_const<T>::type>));
}

int main()
{
    test(3u, "unsigned int");
    // test(boost::serialization::version_type(), "version type");
    return 0;
}
