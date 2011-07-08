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

#ifndef MTL_VPT_VPT_INCLUDE
#define MTL_VPT_VPT_INCLUDE

#ifdef MTL_HAS_VPT
  #include <vt_user.h> 
  #include <string>
  #include <boost/mpl/bool.hpp>
#endif 

namespace mtl { namespace vpt {

#ifdef MTL_HAS_VPT

#ifndef MTL_VPT_LEVEL
#  define MTL_VPT_LEVEL 0
#endif 

template <bool C>
struct sometimes_string
{
    sometimes_string(const char* str) : str(str) {}
    sometimes_string(const std::string& str) : str(str) {}
    std::string str;
};

template <>
struct sometimes_string<false>
{
    sometimes_string(const char*) {}
};

template <int N>
class vampir
{
    typedef boost::mpl::bool_<(MTL_VPT_LEVEL * 100 < N)> to_print;
  public:
    vampir() { entry(to_print());  }

    void entry(boost::mpl::false_) {}
    void entry(boost::mpl::true_) 
    {	
	VT_USER_START(name.str.c_str()); 
	// std::cout << "vpt_entry(" << N << ")\n";    
    }
    
    ~vampir() { end(to_print());  }

    void end(boost::mpl::false_) {}
    void end(boost::mpl::true_) 
    {
	VT_USER_END(name.str.c_str()); 
	// std::cout << "vpt_end(" << N << ")\n";    
    }
    
    bool is_traced() { return to_print::value; }

  private:
    static sometimes_string<to_print::value> name;
};

template <> sometimes_string<(MTL_VPT_LEVEL * 100 < 199)> vampir<199>::name("helper_function");
template <> sometimes_string<(MTL_VPT_LEVEL * 100 < 299)> vampir<299>::name("function");
template <> sometimes_string<(MTL_VPT_LEVEL * 100 < 999)> vampir<999>::name("main");

#else
    template <int N>
    class vampir 
    {
      public:
	vampir() {}
	void show_vpt_level() {}
	bool is_traced() { return false; }
    };
#endif


}} // namespace mtl::vpt

#endif // MTL_VPT_VPT_INCLUDE
