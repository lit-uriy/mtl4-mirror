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

template <int N>
class vampir_trace
{
    typedef boost::mpl::bool_<(MTL_VPT_LEVEL * 100 < N)> to_print;
  public:
    vampir_trace() { entry(to_print());  }

    void entry(boost::mpl::false_) {}
    void entry(boost::mpl::true_) 
    {	
	VT_USER_START(name.c_str()); 
	// std::cout << "vpt_entry(" << N << ")\n";    
    }
    
    ~vampir_trace() { end(to_print());  }

    void end(boost::mpl::false_) {}
    void end(boost::mpl::true_) 
    {
	VT_USER_END(name.c_str()); 
	// std::cout << "vpt_end(" << N << ")\n";    
    }
    
    bool is_traced() { return to_print::value; }

  private:
    static std::string name;
};

template <> std::string vampir_trace<199>::name("helper_function");
template <> std::string vampir_trace<299>::name("function");
template <> std::string vampir_trace<999>::name("main");

#else
    template <int N>
    class vampir_trace 
    {
      public:
	vampir_trace() {}
	void show_vpt_level() {}
	bool is_traced() { return false; }
    };
#endif


} // namespace mtl

using vpt::vampir_trace;

} // namespace mtl

#endif // MTL_VPT_VPT_INCLUDE
