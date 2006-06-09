concept AdditivePartiallyInvertibleMonoid<typename Element>
  : AdditiveMonoid<Element>
{
    where PartiallyInvertibleMonoid< math::add<Element>, Element >;

    // Operator -, binary and unary
    where std::Subtractable<Element>;   
    where Negatable<Element>;

}

auto concept Negatable<typename Element>
{
   typename result_type;
   result_type operator-(Element x);
}


auto concept Divisible<typename T, typename U = T>
{
    typename result_type;
    result_type operator/(T t, U u);
};

#if 0
auto concept DivisibleWithAssign<typename T, typename U = T>
  : Divisible<T, U>
{
    // Operator /= by default defined with /, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator/=(T& x, U y);
#if 0
    {
	return x= x / y;                      defaults NYS
    }
#endif 
}; 
#endif 

auto concept AddableWithAssign<typename T, typename U = T>
  : std::Addable<T, U>
{
    // where std::Addable<T, U>;

    // Operator += by default defined with +, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename assign_result_type;  
    assign_result_type operator+=(T& x, U y);
#if 0
    {
	return x= x + y;                      defaults NYS
    }
#endif 
}; 


auto concept SubtractableWithAssign<typename T, typename U = T>
{
    where std::Subtractable<T, U>;
    
    // Operator -= by default defined with -, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator-=(T& x, U y);
#if 0
    {
	return x= x - y;                      defaults NYS
    }
#endif 
}; 


auto concept MultiplicableWithAssign<typename T, typename U = T>
{
    where std::Multiplicable<T, U>;

    // Operator *= by default defined with *, which is not efficient
    // not efficient, user should implement its own
    // It's not yet supported anyway
    typename result_type;  
    result_type operator*=(T& x, U y);
#if 0
    {
	return x= x * y;                      defaults NYS
    }
#endif 
}; 

