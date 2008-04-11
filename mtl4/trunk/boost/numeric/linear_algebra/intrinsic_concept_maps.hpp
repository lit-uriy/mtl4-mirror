// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MATH_INTRINSIC_CONCEPT_MAPS_INCLUDE
#define MATH_INTRINSIC_CONCEPT_MAPS_INCLUDE

#include <concepts>

namespace math {


    // The following concepts are used to classify intrinsic arithmetic types.
    // The standard concepts already define the syntactic requirements,
    // i.e. the interface.
    // However they sey nothing about the semantics.
    // Therefore, user-defined types can model the syntactic/interface
    // requirements while still having a different mathematical behavior.
    // For that reason, we introduce concepts that are only used for intrinsic types.
    // For them we can define concept_maps regarding semantic behavior as monoids.

    concept IntrinsicSignedIntegral<typename T> 
      : std::SignedIntegralLike<T> 
    {}

    concept IntrinsicUnsignedIntegral<typename T> 
      : std::UnsignedIntegralLike<T> 
    {}

    concept IntrinsicFloatingPoint<typename T>
      : std::FloatingPointLike<T> 
    {}


    // Intrinsic types are chategorized:

    // concept_map IntrinsicSignedIntegral<char> {}; ???
    concept_map IntrinsicSignedIntegral<signed char> {};
    concept_map IntrinsicUnsignedIntegral<unsigned char> {};
    concept_map IntrinsicSignedIntegral<short> {};
    concept_map IntrinsicUnsignedIntegral<unsigned short> {};
    concept_map IntrinsicSignedIntegral<int> {};
    concept_map IntrinsicUnsignedIntegral<unsigned int> {};
    concept_map IntrinsicSignedIntegral<long> {};
    concept_map IntrinsicUnsignedIntegral<unsigned long> {};
    concept_map IntrinsicSignedIntegral<long long> {};
    concept_map IntrinsicUnsignedIntegral<unsigned long long> {};

    concept_map IntrinsicFloatingPoint<float> { }
    concept_map IntrinsicFloatingPoint<double> { }
        

} // namespace math

#endif // MATH_INTRINSIC_CONCEPT_MAPS_INCLUDE
