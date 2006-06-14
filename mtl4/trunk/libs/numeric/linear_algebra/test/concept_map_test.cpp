#include <concepts>

auto concept BinaryIsoFunction<typename Operation, typename Element>
//  : std::Callable2<Operation, Element, Element>
{
   where std::Callable2<Operation, Element, Element>;
    where std::Convertible<std::Callable2<Operation, Element, Element>::result_type, Element>;

    typename result_type = std::Callable2<Operation, Element, Element>::result_type;
};

// auto 
concept Magma<typename Operation, typename Element>
    : BinaryIsoFunction<Operation, Element>
{
    where std::Assignable<Element>;
    where std::Assignable<Element, BinaryIsoFunction<Operation, Element>::result_type>;
};

template <typename Element>
struct add
{
    Element operator() (const Element& x, const Element& y)
    {
	return x + y;
    }
};


// auto 
concept AdditiveMagma<typename Element>
 : Magma< add<Element>, Element >
{
  //  where Magma< add<Element>, Element >;

    typename assign_result_type;  
    assign_result_type operator+=(Element& x, Element y);

    // Operator + is by default defined with +=
    typename result_type;  
    result_type operator+(Element& x, Element y);
#if 0
    {
	Element tmp(x);
	return tmp += y;                      defaults NYS
    }
#endif 
    
    // Type consistency with Magma
    where std::SameType< result_type, 
	                 Magma< add<Element>, Element >::result_type>;

    axiom Consistency(add<Element> op, Element x, Element y)
    {
	op(x, y) == x + y;                    
	// Might change later
        x + y == x += y;
	// Element tmp = x; tmp+= y; tmp == x + y; not proposal-compliant
    }   
}


// concept_map Magma< add<int>, int> {}
concept_map AdditiveMagma<int> {}


int main() {
  return 0;
}
