// $COPYRIGHT$

#ifndef MTL_MAYBE_INCLUDE
#define MTL_MAYBE_INCLUDE

#include <iostream>

namespace mtl { namespace utilities {

template <class Value>
struct maybe : public std::pair<Value, bool> 
{
    typedef std::pair<Value, bool> base; 
    typedef maybe<Value>           self;

    maybe(bool b) : base(Value(), b) {}
    maybe(Value v) : base(v, true) {}
    maybe(Value v, bool b) : base(v, b) {}
    maybe(base b) : base(b) {}

    operator bool() const
    { 
	return this->second; 
    }
    operator Value() const
    { 
	return this->first; 
    }
    bool has_value() const 
    { 
	return this->second; 
    }
    Value value() const 
    { 
	return this->first; 
    }
};

template <class Value>
inline std::ostream& operator<< (std::ostream& os, maybe<Value> const&  m)
{
    return os << '(' << m.value() << ", " << (m ? "true" : "false") << ')';
}

} // namespace utilities

using utilities::maybe; 

} // namespace mtl

#endif // MTL_MAYBE_INCLUDE
