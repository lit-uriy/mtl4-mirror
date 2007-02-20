// $COPYRIGHT$

#ifndef GLAS_GLAS_TAG_INCLUDE
#define GLAS_GLAS_TAG_INCLUDE

namespace glas { namespace tag {

// To iterate only over non-zero elements
struct nz {};

// To iterate over all elements
struct all {};

// To iterate over rows
// Generated cursors must provide range generators
struct row {};

// To iterate over cols
// Generated cursors must provide range generators
struct col {};

// To iterate over the major dimension of matrices (like MTL 2)
struct major {};

// Same with minor
struct minor {};

}} // namespace glas::tag

#endif // GLAS_GLAS_TAG_INCLUDE
