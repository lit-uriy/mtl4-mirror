// $COPYRIGHT$

#ifndef GLAS_GLAS_TAGS_INCLUDE
#define GLAS_GLAS_TAGS_INCLUDE

namespace glas { namespace tags {

// To iterate only over non-zero elements
struct nz_t {};

// To iterate over all elements
struct all_t {};

// To iterate over rows
// Generated cursors must provide range generators
struct row_t {};

// To iterate over cols
// Generated cursors must provide range generators
struct col_t {};

}} // namespace glas::tags

#endif // GLAS_GLAS_TAGS_INCLUDE
